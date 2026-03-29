import os
import math
import json
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import RealPatchDataset
from model import PixelMAE, batch_sample_train_masks, batch_sample_masks, patch_mask_to_pixel_mask
from losses import PixelReconstructionLoss


def log(msg, log_file=None):
    print(msg)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_backend():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    if config.ENABLE_CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    if config.ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def move_to_device(x, device):
    x = x.to(device, non_blocking=True)
    if config.USE_CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def build_model(device):
    model_raw = PixelMAE().to(device)
    if config.USE_CHANNELS_LAST:
        model_raw = model_raw.to(memory_format=torch.channels_last)
    model = model_raw
    if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model_raw)
    return model_raw, model


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def build_optimizer(model_raw):
    return torch.optim.AdamW(
        model_raw.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )


def build_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < config.WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, config.WARMUP_EPOCHS))
        progress = (epoch - config.WARMUP_EPOCHS + 1) / float(max(1, config.EPOCHS - config.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def forward_loss(model, criterion, x, masks):
    pred_patches = model(x, masks)
    pred_img = model.unpatchify(pred_patches)
    pixel_mask = patch_mask_to_pixel_mask(masks, img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE)
    loss, loss_dict = criterion(x, pred_img, pixel_mask)
    return loss, loss_dict


def validate_panel(model, criterion, dl_val, device):
    panel_losses = []
    panel_details = []
    model.eval()
    with torch.no_grad():
        for mask_type, mask_ratio in config.VAL_PANEL:
            loss_sum = 0.0
            raw_sum = 0.0
            lap_sum = 0.0
            count = 0
            for batch in tqdm(dl_val, desc=f"val {mask_type}_{int(mask_ratio * 100)}%", leave=False):
                x = move_to_device(batch, device)
                masks, _ = batch_sample_masks(
                    batch_size=x.shape[0],
                    grid_size=config.GRID_SIZE,
                    mask_ratio=mask_ratio,
                    mask_type=mask_type,
                    device=device,
                )
                loss, loss_dict = forward_loss(model, criterion, x, masks)
                bsz = x.shape[0]
                loss_sum += loss_dict["total"] * bsz
                raw_sum += loss_dict["raw"] * bsz
                lap_sum += loss_dict["lap"] * bsz
                count += bsz
            avg_total = loss_sum / max(1, count)
            avg_raw = raw_sum / max(1, count)
            avg_lap = lap_sum / max(1, count)
            panel_losses.append(avg_total)
            panel_details.append({
                "mask_type": mask_type,
                "mask_ratio": mask_ratio,
                "loss": avg_total,
                "raw": avg_raw,
                "lap": avg_lap,
            })
    val_avg = float(np.mean(panel_losses)) if panel_losses else float("inf")
    return val_avg, panel_details


def train(args):
    set_seed(config.SEED)
    configure_backend()

    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "train_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}", log_file)
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}", log_file)

    ds_train = RealPatchDataset(split="train")
    ds_val = RealPatchDataset(split="val", patches_per_image=1)
    dl_train = make_loader(
        ds_train,
        config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS,
        drop_last=True,
    )
    dl_val = make_loader(
        ds_val,
        config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.VAL_NUM_WORKERS,
        drop_last=False,
    )

    log(f"训练集原图: {len(ds_train.paths)}", log_file)
    log(f"训练集 patch 数: {len(ds_train)}", log_file)
    log(f"验证集原图: {len(ds_val.paths)}", log_file)
    log(
        f"batch={config.TRAIN_BATCH_SIZE}, train_workers={config.TRAIN_NUM_WORKERS}, "
        f"val_workers={config.VAL_NUM_WORKERS}, channels_last={config.USE_CHANNELS_LAST}, "
        f"compile={config.ENABLE_TORCH_COMPILE}",
        log_file,
    )

    model_raw, model = build_model(device)
    criterion = PixelReconstructionLoss(
        lambda_raw=config.LAMBDA_RAW,
        lambda_lap=config.LAMBDA_LAP,
    )
    optimizer = build_optimizer(model_raw)
    scheduler = build_scheduler(optimizer)

    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model_raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        log(f"恢复训练: {args.resume} @ epoch {ckpt['epoch']}", log_file)

    for epoch in range(start_epoch, config.EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_raw = 0.0
        train_lap = 0.0
        train_count = 0
        ratio_meter = []

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for batch in pbar:
            x = move_to_device(batch, device)
            masks, meta = batch_sample_train_masks(x.shape[0], config.GRID_SIZE, device)
            loss, loss_dict = forward_loss(model, criterion, x, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()

            bsz = x.shape[0]
            train_total += loss_dict["total"] * bsz
            train_raw += loss_dict["raw"] * bsz
            train_lap += loss_dict["lap"] * bsz
            train_count += bsz
            ratio_meter.extend([m["actual_ratio"] for m in meta])
            pbar.set_postfix(
                total=f"{loss_dict['total']:.4f}",
                raw=f"{loss_dict['raw']:.4f}",
                lap=f"{loss_dict['lap']:.4f}",
            )

        scheduler.step()
        train_avg = train_total / max(1, train_count)
        train_raw_avg = train_raw / max(1, train_count)
        train_lap_avg = train_lap / max(1, train_count)
        train_ratio_avg = float(np.mean(ratio_meter)) if ratio_meter else 0.0

        val_avg, panel_details = validate_panel(model, criterion, dl_val, device)
        panel_str = " | ".join(
            f"{item['mask_type']}_{int(item['mask_ratio'] * 100)}%={item['loss']:.5f}"
            for item in panel_details
        )
        msg = (
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train_total={train_avg:.6f} raw={train_raw_avg:.6f} lap={train_lap_avg:.6f} "
            f"mask={train_ratio_avg:.4f} | "
            f"val_avg={val_avg:.6f} | {panel_str} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        log(msg, log_file)

        state = {
            "epoch": epoch,
            "model": model_raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "val_avg": val_avg,
            "panel": panel_details,
            "config": {k: v for k, v in config.__dict__.items() if k.isupper()},
        }
        torch.save(state, os.path.join(exp_dir, "last_model.pth"))

        if val_avg < best_val:
            best_val = val_avg
            state["best_val"] = best_val
            torch.save(state, os.path.join(exp_dir, "best_model.pth"))
            log(f"保存最优模型: val_avg={val_avg:.6f}", log_file)

        if epoch % 20 == 0:
            torch.save(state, os.path.join(exp_dir, f"checkpoint_epoch_{epoch}.pth"))

    summary = {
        "best_val": best_val,
        "epochs": config.EPOCHS,
        "exp_name": args.exp_name,
    }
    with open(os.path.join(exp_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"训练完成: best_val={best_val:.6f}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验三 Pixel-space MAE 训练 (5090版)")
    parser.add_argument("--exp_name", type=str, default="Exp3_5090_A0")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)
