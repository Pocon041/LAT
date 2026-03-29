import os
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import kornia

import config
from dataset import _collect_images, _resize_shorter_side, _load_single_image, _preload_images
from model import LatentMAE, sample_train_mask, sample_mask, batch_sample_train_masks, batch_sample_masks
from losses import Stage1Loss, Stage2Loss


# ============================================================
#  Genimage-Tiny 数据集
# ============================================================

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
        paths.extend(_collect_images(nature_dir))
    return paths


class GenimageNatureDataset(Dataset):
    def __init__(self, genimage_root, split="train", generators=None,
                 patches_per_image=8, preload=True, num_threads=16):
        super().__init__()
        self.is_train = (split == "train")
        self.preload = preload

        all_generators = list_generators(genimage_root, selected=generators)
        self.paths = collect_nature_paths(genimage_root, split, all_generators)
        self.patches_per_image = patches_per_image if self.is_train else 1

        if preload:
            resize_short = None if self.is_train else 288
            self.images = _preload_images(
                self.paths,
                resize_short=resize_short,
                num_threads=num_threads,
                desc=f"预加载 genimage_{split}",
            )
        else:
            self.images = None

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        if self.preload and self.images[img_idx] is not None:
            img = self.images[img_idx].copy()
        else:
            img = Image.open(self.paths[img_idx]).convert("RGB")

        if self.is_train:
            w, h = img.size
            if w < config.IMG_SIZE or h < config.IMG_SIZE:
                scale = max(config.IMG_SIZE / w, config.IMG_SIZE / h) + 0.01
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            img = self.train_transform(img)
        else:
            if not self.preload:
                img = _resize_shorter_side(img, target_short=288)
            img = self.val_transform(img)
        return img


# ============================================================
#  通用工具
# ============================================================

def setup_cuda_optimizations():
    if torch.cuda.is_available():
        if getattr(config, "CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print(f"CUDA 优化: benchmark={torch.backends.cudnn.benchmark}, TF32=True, matmul=high")


def get_dataloader(dataset, batch_size, shuffle, num_workers):
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
    return kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)
    )


def log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def resolve_resume_s1(args, exp_dir):
    if args.resume_s1:
        return args.resume_s1
    last = os.path.join(exp_dir, "best_ae.pth")
    return last if os.path.exists(last) else None


def resolve_resume_s2(args, exp_dir):
    if args.resume_s2:
        return args.resume_s2
    last = os.path.join(exp_dir, "checkpoint_s2.pth")
    return last if os.path.exists(last) else None


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
    log(f"genimage_root: {args.genimage_root}", log_file)
    log(f"生成器: {args.generators if args.generators else '全部'}", log_file)

    selected_gens = args.generators if args.generators else None
    log("预加载图像到 RAM...", log_file)
    ds_train = GenimageNatureDataset(
        args.genimage_root, split="train", generators=selected_gens,
        patches_per_image=8, preload=True, num_threads=16,
    )
    ds_val = GenimageNatureDataset(
        args.genimage_root, split="val", generators=selected_gens,
        patches_per_image=1, preload=True, num_threads=16,
    )
    dl_train = get_dataloader(ds_train, config.S1_BATCH_SIZE, shuffle=True, num_workers=config.S1_NUM_WORKERS)
    dl_val = get_dataloader(ds_val, config.S1_BATCH_SIZE, shuffle=False, num_workers=config.S1_NUM_WORKERS)
    log(f"训练集: {len(ds_train.paths)} 张原图, {len(ds_train)} patches", log_file)
    log(f"验证集: {len(ds_val.paths)} 张原图", log_file)

    blur = get_gaussian_blur(config.S1_BLUR_KERNEL_SIZE, config.S1_BLUR_SIGMA).to(device)

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

    start_epoch = 1
    best_val_loss = float("inf")
    resume_path = resolve_resume_s1(args, exp_dir)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.decoder.load_state_dict(ckpt["decoder"])
        if "struct_head" in ckpt:
            model.struct_head.load_state_dict(ckpt["struct_head"])
        if "detail_head" in ckpt:
            model.detail_head.load_state_dict(ckpt["detail_head"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", best_val_loss)
        log(f"恢复阶段一: {resume_path} @ epoch {ckpt.get('epoch', '?')}", log_file)
    else:
        log("阶段一从头开始训练", log_file)

    for epoch in range(start_epoch, config.S1_EPOCHS + 1):
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
            with torch.no_grad():
                x_low = blur(x)
                x_detail_target = x - x_low
            x_recon, z, x_struct_pred, x_detail_pred = model.forward_stage1(x)
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

        model.encoder.eval()
        model.decoder.eval()
        model.struct_head.eval()
        model.detail_head.eval()

        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"[S1] Epoch {epoch} val", leave=False):
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

        msg = (
            f"[S1] Epoch {epoch}/{config.S1_EPOCHS} | "
            f"train: total={train_avg:.6f} struct={train_struct_avg:.6f} detail={train_detail_avg:.6f} | "
            f"val_loss={val_avg:.6f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )
        log(msg, log_file)

        ckpt_state = {
            "epoch": epoch,
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "struct_head": model.struct_head.state_dict(),
            "detail_head": model.detail_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_avg,
        }
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt_state, os.path.join(exp_dir, "best_ae.pth"))
            log(f"  -> 保存最优 AE (val_loss={val_avg:.6f})", log_file)

        if epoch % 20 == 0:
            torch.save(ckpt_state, os.path.join(exp_dir, f"checkpoint_s1_epoch{epoch}.pth"))

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
    log(f"genimage_root: {args.genimage_root}", log_file)
    log(f"生成器: {args.generators if args.generators else '全部'}", log_file)

    selected_gens = args.generators if args.generators else None
    log("预加载图像到 RAM...", log_file)
    ds_train = GenimageNatureDataset(
        args.genimage_root, split="train", generators=selected_gens,
        patches_per_image=8, preload=True, num_threads=16,
    )
    ds_val = GenimageNatureDataset(
        args.genimage_root, split="val", generators=selected_gens,
        patches_per_image=1, preload=True, num_threads=16,
    )
    dl_train = get_dataloader(ds_train, config.S2_BATCH_SIZE, shuffle=True, num_workers=config.S2_NUM_WORKERS)
    dl_val = get_dataloader(ds_val, config.S2_BATCH_SIZE, shuffle=False, num_workers=config.S2_NUM_WORKERS)
    log(f"训练集: {len(ds_train.paths)} 张原图, {len(ds_train)} patches", log_file)
    log(f"验证集: {len(ds_val.paths)} 张原图", log_file)

    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    if getattr(config, "USE_CHANNELS_LAST", False):
        model = model.to(memory_format=torch.channels_last)
        log("已启用 channels_last 内存格式", log_file)

    ae_ckpt_path = args.ae_ckpt if args.ae_ckpt else os.path.join(exp_dir, "best_ae.pth")
    if not os.path.exists(ae_ckpt_path):
        raise FileNotFoundError(f"找不到阶段一权重: {ae_ckpt_path}")
    ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if "struct_head" in ckpt:
        model.struct_head.load_state_dict(ckpt["struct_head"])
    if "detail_head" in ckpt:
        model.detail_head.load_state_dict(ckpt["detail_head"])
    log(f"加载 AE 权重: {ae_ckpt_path} (epoch={ckpt.get('epoch', '?')})", log_file)
    model.freeze_ae()
    log("已冻结 Encoder, Decoder 和辅助头", log_file)

    criterion = Stage2Loss(
        struct_channels=config.STRUCT_CHANNELS,
        lambda_struct=config.S2_LAMBDA_STRUCT,
        struct_l1_weight=config.S2_STRUCT_L1_WEIGHT,
        struct_cos_weight=config.S2_STRUCT_COS_WEIGHT,
        lambda_detail=config.S2_LAMBDA_DETAIL,
        detail_l1_weight=config.S2_DETAIL_L1_WEIGHT,
        detail_cos_weight=config.S2_DETAIL_COS_WEIGHT,
    )
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(), lr=config.S2_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.S2_EPOCHS, eta_min=1e-6
    )

    start_epoch = 1
    best_val_loss = float("inf")
    resume_path = resolve_resume_s2(args, exp_dir)
    if resume_path and os.path.exists(resume_path):
        ckpt_s2 = torch.load(resume_path, map_location=device, weights_only=True)
        model.predictor.load_state_dict(ckpt_s2["predictor"])
        if "optimizer" in ckpt_s2:
            optimizer.load_state_dict(ckpt_s2["optimizer"])
        start_epoch = ckpt_s2.get("epoch", 0) + 1
        best_val_loss = ckpt_s2.get("val_loss", best_val_loss)
        log(f"恢复阶段二: {resume_path} @ epoch {ckpt_s2.get('epoch', '?')}", log_file)
    else:
        log("阶段二从头开始训练", log_file)

    grid_size = config.LATENT_SPATIAL
    val_panel = [
        (0.40, "random"),
        (0.60, "random"),
        (0.75, "random"),
        (0.60, "block"),
    ]

    for epoch in range(start_epoch, config.S2_EPOCHS + 1):
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

        model.predictor.eval()
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
        msg = (
            f"[S2] Epoch {epoch}/{config.S2_EPOCHS} | "
            f"train: total={train_avg:.6f} struct={train_struct_avg:.6f} detail={train_detail_avg:.6f} | "
            f"val_avg={val_avg:.6f} [{panel_str}] | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        log(msg, log_file)

        ckpt_state = {
            "epoch": epoch,
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_avg,
        }
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt_state, os.path.join(exp_dir, "best_predictor.pth"))
            log(f"  -> 保存最优 Predictor (val_loss={val_avg:.6f})", log_file)

        if epoch % 20 == 0:
            torch.save(ckpt_state, os.path.join(exp_dir, "checkpoint_s2.pth"))

    log(f"[阶段二完成] 最优 val_loss={best_val_loss:.6f}", log_file)


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 Genimage-Tiny 真实图像训练 Latent MAE (DC-AE 两阶段)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="训练阶段: 1=AE预训练, 2=Masked Completion")
    parser.add_argument("--exp_name", type=str, default="A0_genimage",
                        help="实验名称")
    parser.add_argument("--genimage_root", type=str, default=r"D:\Genimage-Tiny",
                        help="Genimage-Tiny 根目录")
    parser.add_argument("--generators", nargs="*", default=None,
                        help="指定生成器名列表，默认使用全部")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="阶段二使用的 AE checkpoint (默认同 exp 目录下的 best_ae.pth)")
    parser.add_argument("--resume_s1", type=str, default=None,
                        help="手动指定阶段一续训 checkpoint；不指定则自动检测 best_ae.pth")
    parser.add_argument("--resume_s2", type=str, default=None,
                        help="手动指定阶段二续训 checkpoint；不指定则自动检测 checkpoint_s2.pth")
    args = parser.parse_args()

    if args.stage == 1:
        train_stage1(args)
    else:
        train_stage2(args)
