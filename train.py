import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import RealPatchDataset
from model import LatentMAE, sample_train_mask, sample_mask
from losses import Stage1Loss, Stage2Loss


def log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ============================================================
#  阶段一: AE 预训练 (E + D)
# ============================================================

def train_stage1(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[阶段一] 设备: {device}", log_file)

    # 数据
    ds_train = RealPatchDataset(patches_per_image=8, split="train")
    ds_val = RealPatchDataset(patches_per_image=4, split="val")
    dl_train = DataLoader(
        ds_train, batch_size=config.S1_BATCH_SIZE,
        shuffle=True, num_workers=config.S1_NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=config.S1_BATCH_SIZE,
        shuffle=False, num_workers=config.S1_NUM_WORKERS,
        pin_memory=True,
    )
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)

    # 模型
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    criterion = Stage1Loss(
        lambda_l1=config.S1_LAMBDA_L1,
        lambda_freq=config.S1_LAMBDA_FREQ,
    )

    # 只优化 E + D
    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(ae_params, lr=config.S1_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S1_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    for epoch in range(1, config.S1_EPOCHS + 1):
        # --- 训练 ---
        model.encoder.train()
        model.decoder.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in dl_train:
            x = batch.to(device)
            x_recon, z = model.forward_stage1(x)
            loss, loss_dict = criterion(x, x_recon)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * x.size(0)
            train_count += x.size(0)

        scheduler.step()
        train_avg = train_loss_sum / train_count

        # --- 验证 ---
        model.encoder.eval()
        model.decoder.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in dl_val:
                x = batch.to(device)
                x_recon, z = model.forward_stage1(x)
                loss, loss_dict = criterion(x, x_recon)
                val_loss_sum += loss_dict["total"] * x.size(0)
                val_count += x.size(0)

        val_avg = val_loss_sum / val_count

        msg = (f"[S1] Epoch {epoch}/{config.S1_EPOCHS} | "
               f"train_loss={train_avg:.6f} | val_loss={val_avg:.6f} | "
               f"lr={scheduler.get_last_lr()[0]:.2e}")
        log(msg, log_file)

        # 保存最优
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "best_ae.pth"))
            log(f"  -> 保存最优 AE (val_loss={val_avg:.6f})", log_file)

        # 定期保存 checkpoint
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "predictor": model.predictor.state_dict(),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[阶段二] 设备: {device}", log_file)

    # 数据
    ds_train = RealPatchDataset(patches_per_image=8, split="train")
    ds_val = RealPatchDataset(patches_per_image=4, split="val")
    dl_train = DataLoader(
        ds_train, batch_size=config.S2_BATCH_SIZE,
        shuffle=True, num_workers=config.S2_NUM_WORKERS,
        pin_memory=True, drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=config.S2_BATCH_SIZE,
        shuffle=False, num_workers=config.S2_NUM_WORKERS,
        pin_memory=True,
    )
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)

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

    # 冻结 E 和 D
    model.freeze_ae()
    log("已冻结 Encoder 和 Decoder", log_file)

    # 损失
    criterion = Stage2Loss(
        lambda_l1=config.S2_LAMBDA_L1,
        lambda_cos=config.S2_LAMBDA_COS,
    )

    # 只优化 P
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(), lr=config.S2_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S2_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    grid_size = config.LATENT_SPATIAL

    for epoch in range(1, config.S2_EPOCHS + 1):
        # --- 训练 ---
        model.predictor.train()
        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_cos_sum = 0.0
        train_count = 0

        for batch in dl_train:
            x = batch.to(device)
            B = x.size(0)

            # 为 batch 中每个样本独立采样 mask
            masks = []
            for _ in range(B):
                m, _, _ = sample_train_mask(grid_size)
                masks.append(m)
            masks = torch.stack(masks).to(device)  # [B, N]

            z, z_hat = model.forward_stage2(x, masks)
            loss, loss_dict = criterion(z, z_hat, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * B
            train_l1_sum += loss_dict["l1"] * B
            train_cos_sum += loss_dict["cos"] * B
            train_count += B

        scheduler.step()
        train_avg = train_loss_sum / train_count
        train_l1_avg = train_l1_sum / train_count
        train_cos_avg = train_cos_sum / train_count

        # --- 验证: 多条件面板平均 ---
        model.predictor.eval()
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
                for batch in dl_val:
                    x = batch.to(device)
                    B = x.size(0)
                    masks = torch.stack([
                        sample_mask(grid_size, p_ratio, p_type)[0] for _ in range(B)
                    ]).to(device)

                    z, z_hat = model.forward_stage2(x, masks)
                    loss, loss_dict = criterion(z, z_hat, masks)
                    vl_sum += loss_dict["total"] * B
                    vl_count += B
                panel_losses.append(vl_sum / vl_count)

        val_avg = sum(panel_losses) / len(panel_losses)
        panel_str = " | ".join(
            f"{t}_{int(r*100)}%={l:.6f}" for (r, t), l in zip(val_panel, panel_losses)
        )

        msg = (f"[S2] Epoch {epoch}/{config.S2_EPOCHS} | "
               f"train: total={train_avg:.6f} l1={train_l1_avg:.6f} cos={train_cos_avg:.6f} | "
               f"val_avg={val_avg:.6f} [{panel_str}] | "
               f"lr={scheduler.get_last_lr()[0]:.2e}")
        log(msg, log_file)

        # 保存最优
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "best_predictor.pth"))
            log(f"  -> 保存最优 Predictor (val_loss={val_avg:.6f})", log_file)

        # 定期 checkpoint
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_avg,
            }, os.path.join(exp_dir, "checkpoint_s2.pth"))

    log(f"[阶段二完成] 最优 val_loss={best_val_loss:.6f}", log_file)


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE 训练")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="训练阶段: 1=AE预训练, 2=Masked Completion")
    parser.add_argument("--exp_name", type=str, default="A0",
                        help="实验名称, 用于创建输出目录")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="阶段二使用的 AE checkpoint 路径 (默认使用同目录下的 best_ae.pth)")
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)
