import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as config
from dataset import RealPatchDataset
from model import LatentMAE, sample_train_mask, sample_mask
from losses import Stage1Loss, Stage2Loss


def log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    """统一构建 DataLoader, 启用 5090 优化选项"""
    persistent = config.PERSISTENT_WORKERS and num_workers > 0
    prefetch = config.PREFETCH_FACTOR if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=shuffle,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def setup_device():
    """初始化设备 + cudnn 优化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.CUDNN_BENCHMARK and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


# ============================================================
#  阶段一: AE 预训练 (E + D)
# ============================================================

def train_stage1(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train_log.txt")

    device = setup_device()
    log(f"[阶段一] 设备: {device}, cudnn.benchmark={torch.backends.cudnn.benchmark}", log_file)

    # 数据
    ds_train = RealPatchDataset(patches_per_image=8, split="train")
    ds_val = RealPatchDataset(split="val")
    dl_train = make_dataloader(ds_train, config.S1_BATCH_SIZE, shuffle=True, num_workers=config.S1_NUM_WORKERS)
    dl_val = make_dataloader(ds_val, config.S1_BATCH_SIZE, shuffle=False, num_workers=config.S1_NUM_WORKERS)
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)
    log(f"batch_size={config.S1_BATCH_SIZE}, workers={config.S1_NUM_WORKERS}, "
        f"preload={getattr(config, 'PRELOAD_TO_RAM', False)}", log_file)

    # 模型
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    if config.TORCH_COMPILE:
        model = torch.compile(model)
        log("已启用 torch.compile", log_file)
    criterion = Stage1Loss(
        lambda_l1=config.S1_LAMBDA_L1,
        lambda_freq=config.S1_LAMBDA_FREQ,
    )

    # 只优化 E + D
    ae_params = list(model.parameters())  # compile 后直接用 model.parameters()
    optimizer = torch.optim.AdamW(ae_params, lr=config.S1_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S1_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    for epoch in range(1, config.S1_EPOCHS + 1):
        t0 = time.time()
        # --- 训练 ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(dl_train, desc=f"[S1] Epoch {epoch}/{config.S1_EPOCHS} train")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            x_recon, z = model.forward_stage1(x)
            loss, loss_dict = criterion(x, x_recon)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * x.size(0)
            train_count += x.size(0)
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}")

        scheduler.step()
        train_avg = train_loss_sum / train_count

        # --- 验证 ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"[S1] Epoch {epoch}/{config.S1_EPOCHS} val"):
                x = batch.to(device, non_blocking=True)
                x_recon, z = model.forward_stage1(x)
                loss, loss_dict = criterion(x, x_recon)
                val_loss_sum += loss_dict["total"] * x.size(0)
                val_count += x.size(0)

        val_avg = val_loss_sum / val_count
        elapsed = time.time() - t0

        msg = (f"[S1] Epoch {epoch}/{config.S1_EPOCHS} | "
               f"train_loss={train_avg:.6f} | val_loss={val_avg:.6f} | "
               f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        log(msg, log_file)

        # 保存最优
        # compile 后需要用 _orig_mod 取原始模型
        raw_model = getattr(model, "_orig_mod", model)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "encoder": raw_model.encoder.state_dict(),
                "decoder": raw_model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "best_ae.pth"))
            log(f"  -> 保存最优 AE (val_loss={val_avg:.6f})", log_file)

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "encoder": raw_model.encoder.state_dict(),
                "decoder": raw_model.decoder.state_dict(),
                "predictor": raw_model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "checkpoint.pth"))

    log(f"[阶段一完成] 最优 val_loss={best_val_loss:.6f}", log_file)


# ============================================================
#  阶段二: Masked Latent Completion (只训练 P)
# ============================================================

def train_stage2(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train_log.txt")

    device = setup_device()
    log(f"[阶段二] 设备: {device}, cudnn.benchmark={torch.backends.cudnn.benchmark}", log_file)

    # 数据
    ds_train = RealPatchDataset(patches_per_image=8, split="train")
    ds_val = RealPatchDataset(split="val")
    dl_train = make_dataloader(ds_train, config.S2_BATCH_SIZE, shuffle=True, num_workers=config.S2_NUM_WORKERS)
    dl_val = make_dataloader(ds_val, config.S2_BATCH_SIZE, shuffle=False, num_workers=config.S2_NUM_WORKERS)
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)
    log(f"batch_size={config.S2_BATCH_SIZE}, workers={config.S2_NUM_WORKERS}, "
        f"preload={getattr(config, 'PRELOAD_TO_RAM', False)}", log_file)

    # 模型: 加载阶段一权重
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)

    ae_ckpt_path = args.ae_ckpt
    if ae_ckpt_path is None:
        ae_ckpt_path = os.path.join(exp_dir, "best_ae.pth")
    if not os.path.exists(ae_ckpt_path):
        raise FileNotFoundError(f"找不到阶段一权重: {ae_ckpt_path}")

    ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    log(f"加载 AE 权重: {ae_ckpt_path} (epoch={ckpt['epoch']})", log_file)

    model.freeze_ae()
    log("已冻结 Encoder 和 Decoder", log_file)

    if config.TORCH_COMPILE:
        model = torch.compile(model)
        log("已启用 torch.compile", log_file)

    criterion = Stage2Loss(
        lambda_l1=config.S2_LAMBDA_L1,
        lambda_cos=config.S2_LAMBDA_COS,
    )

    # 只优化 P
    raw_model = getattr(model, "_orig_mod", model)
    optimizer = torch.optim.AdamW(
        raw_model.predictor.parameters(), lr=config.S2_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S2_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    grid_size = config.LATENT_SPATIAL

    for epoch in range(1, config.S2_EPOCHS + 1):
        t0 = time.time()
        # --- 训练 ---
        raw_model.predictor.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_cos_sum = 0.0
        train_count = 0

        pbar = tqdm(dl_train, desc=f"[S2] Epoch {epoch}/{config.S2_EPOCHS} train")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            B = x.size(0)

            masks = []
            for _ in range(B):
                m, _, _ = sample_train_mask(grid_size)
                masks.append(m)
            masks = torch.stack(masks).to(device, non_blocking=True)

            z, z_hat = model.forward_stage2(x, masks)
            loss, loss_dict = criterion(z, z_hat, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * B
            train_l1_sum += loss_dict["l1"] * B
            train_cos_sum += loss_dict["cos"] * B
            train_count += B
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}")

        scheduler.step()
        train_avg = train_loss_sum / train_count
        train_l1_avg = train_l1_sum / train_count
        train_cos_avg = train_cos_sum / train_count

        # --- 验证: 多条件面板平均 ---
        raw_model.predictor.eval()
        val_panel = [
            (0.30, "random"),
            (0.50, "random"),
            (0.60, "random"),
            (0.50, "block"),
        ]
        panel_losses = []

        with torch.no_grad():
            for p_ratio, p_type in val_panel:
                vl_sum = 0.0
                vl_count = 0
                for batch in tqdm(dl_val, desc=f"  val {p_type}_{int(p_ratio*100)}%", leave=False):
                    x = batch.to(device, non_blocking=True)
                    B = x.size(0)
                    masks = torch.stack([
                        sample_mask(grid_size, p_ratio, p_type)[0] for _ in range(B)
                    ]).to(device, non_blocking=True)

                    z, z_hat = model.forward_stage2(x, masks)
                    loss, loss_dict = criterion(z, z_hat, masks)
                    vl_sum += loss_dict["total"] * B
                    vl_count += B
                panel_losses.append(vl_sum / vl_count)

        val_avg = sum(panel_losses) / len(panel_losses)
        panel_str = " | ".join(
            f"{t}_{int(r*100)}%={l:.6f}" for (r, t), l in zip(val_panel, panel_losses)
        )
        elapsed = time.time() - t0

        msg = (f"[S2] Epoch {epoch}/{config.S2_EPOCHS} | "
               f"train: total={train_avg:.6f} l1={train_l1_avg:.6f} cos={train_cos_avg:.6f} | "
               f"val_avg={val_avg:.6f} [{panel_str}] | "
               f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")
        log(msg, log_file)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "predictor": raw_model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "best_predictor.pth"))
            log(f"  -> 保存最优 Predictor (val_loss={val_avg:.6f})", log_file)

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "predictor": raw_model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "checkpoint_s2.pth"))

    log(f"[阶段二完成] 最优 val_loss={best_val_loss:.6f}", log_file)


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE 训练 (5090 优化版)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="训练阶段: 1=AE预训练, 2=Masked Completion")
    parser.add_argument("--exp_name", type=str, default="A0",
                        help="实验名称, 用于创建输出目录")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="阶段二使用的 AE checkpoint 路径")
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)
