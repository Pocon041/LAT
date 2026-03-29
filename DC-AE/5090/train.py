import os
os.environ["OMP_NUM_THREADS"] = "4"
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia

import config
from dataset import RealPatchDataset
from model import LatentMAE, sample_train_mask, sample_mask, batch_sample_train_masks, batch_sample_masks
from losses import Stage1Loss, Stage2Loss


def setup_cuda_optimizations():
    """设置 CUDA 优化"""
    if torch.cuda.is_available():
        if getattr(config, "CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print(f"CUDA 优化: benchmark={torch.backends.cudnn.benchmark}, TF32=True, matmul=high")


def get_dataloader(dataset, batch_size, shuffle, num_workers):
    """创建优化的 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=getattr(config, "PIN_MEMORY", True),
        prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2),
        persistent_workers=getattr(config, "PERSISTENT_WORKERS", False) and num_workers > 0,
        drop_last=shuffle,
    )


def get_gaussian_blur(kernel_size, sigma):
    """创建高斯模糊层, 用于生成 X_low"""
    return kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)
    )


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
    setup_cuda_optimizations()
    log(f"[阶段一] 设备: {device}, batch_size={config.S1_BATCH_SIZE}", log_file)

    # 数据 (RAM 预加载 + 优化 DataLoader)
    log("预加载图像到 RAM...", log_file)
    ds_train = RealPatchDataset(patches_per_image=8, split="train", preload=True, num_threads=16)
    ds_val = RealPatchDataset(split="val", preload=True, num_threads=16)
    dl_train = get_dataloader(ds_train, config.S1_BATCH_SIZE, shuffle=True, num_workers=config.S1_NUM_WORKERS)
    dl_val = get_dataloader(ds_val, config.S1_BATCH_SIZE, shuffle=False, num_workers=config.S1_NUM_WORKERS)
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)
    log(f"DataLoader: workers={config.S1_NUM_WORKERS}, prefetch={getattr(config, 'PREFETCH_FACTOR', 2)}", log_file)

    # 高斯模糊层: 生成 X_low
    blur = get_gaussian_blur(
        config.S1_BLUR_KERNEL_SIZE, config.S1_BLUR_SIGMA
    ).to(device)
    log(f"高斯模糊: kernel={config.S1_BLUR_KERNEL_SIZE}, sigma={config.S1_BLUR_SIGMA}", log_file)

    # 模型
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    if getattr(config, "USE_CHANNELS_LAST", False):
        model = model.to(memory_format=torch.channels_last)
        log("已启用 channels_last 内存格式", log_file)
    criterion = Stage1Loss(
        lambda_l1=config.S1_LAMBDA_L1,
        lambda_freq=config.S1_LAMBDA_FREQ,
        lambda_struct=config.S1_LAMBDA_STRUCT,
        struct_l1_weight=config.S1_STRUCT_L1_WEIGHT,
        struct_cos_weight=config.S1_STRUCT_COS_WEIGHT,
        lambda_detail=config.S1_LAMBDA_DETAIL,
    )

    # 优化 E + D + H_s + H_d
    ae_params = (
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.struct_head.parameters()) +
        list(model.detail_head.parameters())
    )
    optimizer = torch.optim.AdamW(ae_params, lr=config.S1_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S1_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    for epoch in range(1, config.S1_EPOCHS + 1):
        # --- 训练 ---
        model.encoder.train()
        model.decoder.train()
        model.struct_head.train()
        model.detail_head.train()
        
        train_loss_sum = 0.0
        train_struct_sum = 0.0
        train_detail_sum = 0.0
        train_count = 0

        pbar = tqdm(dl_train, desc=f"[S1] Epoch {epoch}/{config.S1_EPOCHS} train")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            if getattr(config, "USE_CHANNELS_LAST", False):
                x = x.to(memory_format=torch.channels_last)
            
            # 生成 X_low 和 X_detail_target
            with torch.no_grad():
                x_low = blur(x)
                x_detail_target = x - x_low
            
            # 前向
            x_recon, z, x_struct_pred, x_detail_pred = model.forward_stage1(x)
            
            # 损失
            loss, loss_dict = criterion(
                x, x_recon, x_low, x_struct_pred, x_detail_target, x_detail_pred
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_params, max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * x.size(0)
            train_struct_sum += loss_dict["struct"] * x.size(0)
            train_detail_sum += loss_dict["detail"] * x.size(0)
            train_count += x.size(0)
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                struct=f"{loss_dict['struct']:.4f}",
                detail=f"{loss_dict['detail']:.4f}",
            )

        scheduler.step()
        train_avg = train_loss_sum / train_count
        train_struct_avg = train_struct_sum / train_count
        train_detail_avg = train_detail_sum / train_count

        # --- 验证 ---
        model.encoder.eval()
        model.decoder.eval()
        model.struct_head.eval()
        model.detail_head.eval()
        
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"[S1] Epoch {epoch}/{config.S1_EPOCHS} val"):
                x = batch.to(device, non_blocking=True)
                if getattr(config, "USE_CHANNELS_LAST", False):
                    x = x.to(memory_format=torch.channels_last)
                x_low = blur(x)
                x_detail_target = x - x_low
                x_recon, z, x_struct_pred, x_detail_pred = model.forward_stage1(x)
                loss, loss_dict = criterion(
                    x, x_recon, x_low, x_struct_pred, x_detail_target, x_detail_pred
                )
                val_loss_sum += loss_dict["total"] * x.size(0)
                val_count += x.size(0)

        val_avg = val_loss_sum / val_count

        msg = (f"[S1] Epoch {epoch}/{config.S1_EPOCHS} | "
               f"train: total={train_avg:.6f} struct={train_struct_avg:.6f} detail={train_detail_avg:.6f} | "
               f"val_loss={val_avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")
        log(msg, log_file)

        # 保存最优
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save({
                "epoch": epoch,
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "struct_head": model.struct_head.state_dict(),
                "detail_head": model.detail_head.state_dict(),
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
                "struct_head": model.struct_head.state_dict(),
                "detail_head": model.detail_head.state_dict(),
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
    setup_cuda_optimizations()
    log(f"[阶段二] 设备: {device}, batch_size={config.S2_BATCH_SIZE}", log_file)

    # 数据 (RAM 预加载 + 优化 DataLoader)
    log("预加载图像到 RAM...", log_file)
    ds_train = RealPatchDataset(patches_per_image=8, split="train", preload=True, num_threads=16)
    ds_val = RealPatchDataset(split="val", preload=True, num_threads=16)
    dl_train = get_dataloader(ds_train, config.S2_BATCH_SIZE, shuffle=True, num_workers=config.S2_NUM_WORKERS)
    dl_val = get_dataloader(ds_val, config.S2_BATCH_SIZE, shuffle=False, num_workers=config.S2_NUM_WORKERS)
    log(f"训练集: {len(ds_train)} patches, 验证集: {len(ds_val)} patches", log_file)
    log(f"DataLoader: workers={config.S2_NUM_WORKERS}, prefetch={getattr(config, 'PREFETCH_FACTOR', 2)}", log_file)

    # 模型: 加载阶段一权重
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    if getattr(config, "USE_CHANNELS_LAST", False):
        model = model.to(memory_format=torch.channels_last)
        log("已启用 channels_last 内存格式", log_file)

    ae_ckpt_path = args.ae_ckpt
    if ae_ckpt_path is None:
        ae_ckpt_path = os.path.join(exp_dir, "best_ae.pth")
    if not os.path.exists(ae_ckpt_path):
        raise FileNotFoundError(f"找不到阶段一权重: {ae_ckpt_path}")

    ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    # 加载辅助头 (如果存在)
    if "struct_head" in ckpt:
        model.struct_head.load_state_dict(ckpt["struct_head"])
    if "detail_head" in ckpt:
        model.detail_head.load_state_dict(ckpt["detail_head"])
    log(f"加载 AE 权重: {ae_ckpt_path} (epoch={ckpt['epoch']})", log_file)

    # 冻结 E 和 D (含辅助头)
    model.freeze_ae()
    log("已冻结 Encoder, Decoder 和辅助头", log_file)

    # 损失 (分通道)
    criterion = Stage2Loss(
        struct_channels=config.STRUCT_CHANNELS,
        lambda_struct=config.S2_LAMBDA_STRUCT,
        struct_l1_weight=config.S2_STRUCT_L1_WEIGHT,
        struct_cos_weight=config.S2_STRUCT_COS_WEIGHT,
        lambda_detail=config.S2_LAMBDA_DETAIL,
        detail_l1_weight=config.S2_DETAIL_L1_WEIGHT,
        detail_cos_weight=config.S2_DETAIL_COS_WEIGHT,
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
        train_struct_sum = 0.0
        train_detail_sum = 0.0
        train_count = 0

        pbar = tqdm(dl_train, desc=f"[S2] Epoch {epoch}/{config.S2_EPOCHS} train")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            if getattr(config, "USE_CHANNELS_LAST", False):
                x = x.to(memory_format=torch.channels_last)
            B = x.size(0)
            masks = batch_sample_train_masks(B, grid_size, device)

            z, z_hat = model.forward_stage2(x, masks)
            loss, loss_dict = criterion(z, z_hat, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss_dict["total"] * B
            train_struct_sum += loss_dict["struct"] * B
            train_detail_sum += loss_dict["detail"] * B
            train_count += B
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.4f}",
                struct=f"{loss_dict['struct']:.4f}",
                detail=f"{loss_dict['detail']:.4f}",
            )

        scheduler.step()
        train_avg = train_loss_sum / train_count
        train_struct_avg = train_struct_sum / train_count
        train_detail_avg = train_detail_sum / train_count

        # --- 验证: 多条件面板平均 ---
        model.predictor.eval()
        val_panel = [
            (0.40, "random"),
            (0.60, "random"),
            (0.75, "random"),
            (0.60, "block"),
        ]
        panel_losses = []

        with torch.no_grad():
            for p_ratio, p_type in val_panel:
                vl_sum = 0.0
                vl_count = 0
                for batch in tqdm(dl_val, desc=f"  val {p_type}_{int(p_ratio*100)}%", leave=False):
                    x = batch.to(device, non_blocking=True)
                    if getattr(config, "USE_CHANNELS_LAST", False):
                        x = x.to(memory_format=torch.channels_last)
                    B = x.size(0)
                    masks = batch_sample_masks(B, grid_size, p_ratio, p_type, device)

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
               f"train: total={train_avg:.6f} struct={train_struct_avg:.6f} detail={train_detail_avg:.6f} | "
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
