import os
import math
import json
import random
import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import config
from dataset import collect_images, resize_shorter_side, ensure_min_size
from model import PixelMAE, batch_sample_train_masks, batch_sample_masks, patch_mask_to_pixel_mask
from losses import PixelReconstructionLoss


def load_image_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB").copy()


def preload_images(paths, desc, num_workers=4):
    if not paths:
        return []
    print(f"[预加载] {desc}: {len(paths)} 张")
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_image_rgb, p): i for i, p in enumerate(paths)}
        for future in tqdm(futures, total=len(futures), desc=f"预加载 {desc}"):
            results[futures[future]] = future.result()
    valid = sum(r is not None for r in results)
    print(f"[预加载] 完成: {valid}/{len(paths)}")
    return results


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
    torch.backends.cudnn.benchmark = True
    if config.ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def list_generators(genimage_root, selected=None):
    selected_set = set(selected) if selected else None
    result = []
    for name in sorted(os.listdir(genimage_root)):
        path = os.path.join(genimage_root, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isdir(os.path.join(path, "train")):
            continue
        if selected_set is not None and name not in selected_set:
            continue
        result.append((name, path))
    return result


def collect_nature_paths(genimage_root, split, generators):
    paths = []
    for gen_name, gen_path in generators:
        nature_dir = os.path.join(gen_path, split, "nature")
        gen_paths = collect_images(nature_dir)
        paths.extend(gen_paths)
    return paths


class GenimageNatureDataset(Dataset):
    def __init__(self, genimage_root, split="train", generators=None,
                 patches_per_image=None, preload=False):
        super().__init__()
        self.split = split
        self.is_train = split == "train"
        self.patches_per_image = patches_per_image if patches_per_image is not None else config.PATCHES_PER_IMAGE

        all_generators = list_generators(genimage_root, selected=generators)
        self.paths = collect_nature_paths(genimage_root, split, all_generators)

        self.preload = preload
        self.images = preload_images(self.paths, f"genimage_{split}") if preload else None

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.eval_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def get_image(self, img_idx):
        if self.images is not None:
            return self.images[img_idx].copy()
        return load_image_rgb(self.paths[img_idx])

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = self.get_image(img_idx)
        if self.is_train:
            img = ensure_min_size(img, config.IMG_SIZE)
            img = self.train_transform(img)
        else:
            img = resize_shorter_side(img, target_short=288)
            img = ensure_min_size(img, config.IMG_SIZE)
            img = self.eval_transform(img)
        return img


def make_loader(dataset, batch_size, shuffle, drop_last=False):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.PIN_MEMORY,
        "drop_last": drop_last,
    }
    if config.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def build_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
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
                x = batch.to(device, non_blocking=True)
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


def resolve_resume(args, exp_dir):
    if args.resume:
        return args.resume
    last_ckpt = os.path.join(exp_dir, "last_model.pth")
    if os.path.exists(last_ckpt):
        return last_ckpt
    return None


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

    selected_gens = args.generators if args.generators else None
    ds_train = GenimageNatureDataset(
        args.genimage_root, split="train",
        generators=selected_gens,
        patches_per_image=config.PATCHES_PER_IMAGE,
        preload=args.preload_train,
    )
    ds_val = GenimageNatureDataset(
        args.genimage_root, split="val",
        generators=selected_gens,
        patches_per_image=1,
        preload=args.preload_val,
    )
    dl_train = make_loader(ds_train, config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val = make_loader(ds_val, config.VAL_BATCH_SIZE, shuffle=False, drop_last=False)

    log(f"训练集原图: {len(ds_train.paths)}", log_file)
    log(f"训练集 patch 数: {len(ds_train)}", log_file)
    log(f"验证集原图: {len(ds_val.paths)}", log_file)
    log(f"生成器列表: {list_generators(args.genimage_root, selected=selected_gens)}", log_file)

    model = PixelMAE().to(device)
    criterion = PixelReconstructionLoss(
        lambda_raw=config.LAMBDA_RAW,
        lambda_lap=config.LAMBDA_LAP,
    )
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    start_epoch = 1
    best_val = float("inf")
    resume_path = resolve_resume(args, exp_dir)
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        log(f"恢复训练: {resume_path} @ epoch {ckpt['epoch']}", log_file)
    else:
        log("从头开始训练", log_file)

    for epoch in range(start_epoch, config.EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_raw = 0.0
        train_lap = 0.0
        train_count = 0
        ratio_meter = []

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for batch in pbar:
            x = batch.to(device, non_blocking=True)
            masks, meta = batch_sample_train_masks(x.shape[0], config.GRID_SIZE, device)
            loss, loss_dict = forward_loss(model, criterion, x, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
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
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "val_avg": val_avg,
            "panel": panel_details,
            "genimage_root": args.genimage_root,
            "generators": selected_gens,
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
        "genimage_root": args.genimage_root,
        "generators": selected_gens,
    }
    with open(os.path.join(exp_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"训练完成: best_val={best_val:.6f}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 Genimage-Tiny 真实图像训练 Pixel-space MAE")
    parser.add_argument("--exp_name", type=str, default="Exp3_genimage_A0")
    parser.add_argument("--genimage_root", type=str, default=r"D:\Genimage-Tiny")
    parser.add_argument("--generators", nargs="*", default=None,
                        help="指定生成器名列表，默认使用全部")
    parser.add_argument("--resume", type=str, default=None,
                        help="手动指定 checkpoint 路径；不指定则自动检测 last_model.pth")
    parser.add_argument("--preload_train", action="store_true",
                        help="训练集预加载到 RAM")
    parser.add_argument("--preload_val", action="store_true",
                        help="验证集预加载到 RAM")
    args = parser.parse_args()
    train(args)
