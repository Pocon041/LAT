"""
Latent MAE 双路径训练
支持四种模式:
  A0: plain     (无 mask, alpha=0)
  A1: keep=0.7  (固定 keep_ratio=0.7)
  A2: keep=0.5  (固定 keep_ratio=0.5)
  A3: curriculum (keep_ratio 从 0.9 线性退到 0.5)

用法:
  python train.py --mode A0
  python train.py --mode A1
  python train.py --mode A2
  python train.py --mode A3
"""
import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import SimpleAE
from dataset import RealTrainDataset, RealValDataset
from losses import CombinedLoss, LightLoss


def get_keep_ratio(mode, epoch, total_epochs):
    """根据模式和当前 epoch 返回 keep_ratio, None 表示不做 mask"""
    if mode == "A0":
        return None
    elif mode == "A1":
        return 0.7
    elif mode == "A2":
        return 0.5
    elif mode == "A3":
        # curriculum: 0.9 -> 0.5, 在前 2/3 epochs 线性退火, 之后固定 0.5
        warmdown_epochs = int(total_epochs * 2 / 3)
        if epoch <= warmdown_epochs:
            t = epoch / warmdown_epochs
            return 0.9 - 0.4 * t
        else:
            return 0.5
    else:
        raise ValueError(f"未知模式: {mode}")


def train(mode, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}", flush=True)
    print(f"模式: {mode}", flush=True)

    # 输出目录
    exp_dir = os.path.join(config.EXP_ROOT, mode)
    os.makedirs(exp_dir, exist_ok=True)

    # 数据
    train_ds = RealTrainDataset()
    val_ds = RealValDataset()
    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # 模型
    model = SimpleAE(latent_channels=config.LATENT_CHANNELS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}", flush=True)

    # 损失和优化器
    criterion = CombinedLoss(
        lambda_l1=config.LAMBDA_L1,
        lambda_lpips=config.LAMBDA_LPIPS,
        lambda_freq=config.LAMBDA_FREQ,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # masked 路径用轻量损失 (L1+Freq, 无 LPIPS), 加速训练
    criterion_mask = LightLoss(
        lambda_l1=config.LAMBDA_L1,
        lambda_freq=config.LAMBDA_FREQ,
    )
    alpha = config.MASK_ALPHA
    best_val_loss = float("inf")
    start_epoch = 1
    log_path = os.path.join(exp_dir, "train_log.txt")
    ckpt_path = os.path.join(exp_dir, "checkpoint.pth")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 断点续训
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            log(f"从 epoch {start_epoch} 恢复训练 (best_val={best_val_loss:.4f})")
        else:
            # 兼容旧格式: checkpoint 只有 state_dict
            model.load_state_dict(ckpt)
            log(f"从旧格式 checkpoint 加载模型权重, 从 epoch 1 重新开始训练")
    else:
        log(f"训练开始 | mode={mode}, epochs={config.EPOCHS}, batch={config.BATCH_SIZE}, lr={config.LR}")
        log(f"损失权重: L1={config.LAMBDA_L1}, LPIPS={config.LAMBDA_LPIPS}, Freq={config.LAMBDA_FREQ}, alpha={alpha}")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        t0 = time.time()
        kr = get_keep_ratio(mode, epoch, config.EPOCHS)

        # --- Train ---
        model.train()
        train_losses = {"full": 0, "mask": 0, "total": 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False, file=sys.stdout)
        for batch in pbar:
            x = batch.to(device)

            if kr is not None:
                recon_full, recon_mask, z = model(x, mask_keep_ratio=kr)
                loss_full, _ = criterion(x, recon_full)
                loss_mask, _ = criterion_mask(x, recon_mask)
                loss = loss_full + alpha * loss_mask
                train_losses["full"] += loss_full.item()
                train_losses["mask"] += loss_mask.item()
            else:
                recon_full, z = model(x)
                loss, _ = criterion(x, recon_full)
                train_losses["full"] += loss.item()

            train_losses["total"] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_batches = len(train_loader)
        for k in train_losses:
            train_losses[k] /= n_batches

        # --- Val (full 路径) ---
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} val  ", leave=False, file=sys.stdout):
                x = batch.to(device)
                recon_full, z = model(x)
                loss_val, _ = criterion(x, recon_full)
                val_loss_total += loss_val.item()
        val_loss = val_loss_total / max(len(val_loader), 1)

        scheduler.step()
        elapsed = time.time() - t0

        kr_str = f"{kr:.2f}" if kr is not None else "none"
        msg = (f"[Epoch {epoch:03d}/{config.EPOCHS}] "
               f"kr={kr_str} "
               f"train: full={train_losses['full']:.4f} mask={train_losses['mask']:.4f} "
               f"total={train_losses['total']:.4f} | "
               f"val={val_loss:.4f} | {elapsed:.1f}s")
        log(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_ae.pth"))
            log(f"  -> 保存最优模型 (val={best_val_loss:.4f})")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(exp_dir, f"ae_epoch{epoch}.pth"))

        # 每个 epoch 保存 checkpoint 用于断点续训
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_path)

    log(f"训练结束, 最优 val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["A0", "A1", "A2", "A3"])
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复训练")
    args = parser.parse_args()
    train(args.mode, resume=args.resume)
