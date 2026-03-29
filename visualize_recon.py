import os
import math
import random
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from dataset import ChameleonTestDataset, RealPatchDataset
from model import PixelMAE, sample_mask, patch_mask_to_pixel_mask, erode_pixel_mask
from losses import laplacian_filter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(exp_dir, device):
    model = PixelMAE().to(device)
    best_path = os.path.join(exp_dir, "best_model.pth")
    last_path = os.path.join(exp_dir, "last_model.pth")
    ckpt_path = best_path if os.path.exists(best_path) else last_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型权重: {best_path} 或 {last_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_path, ckpt.get("epoch", -1)


def tensor_to_rgb(x):
    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def to_heatmap(x, vmin=None, vmax=None):
    arr = x.detach().cpu().numpy()
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if math.isclose(vmax, vmin):
        vmax = vmin + 1e-6
    return arr, vmin, vmax


def compute_error_maps(x, recon):
    raw_map = (x - recon).abs().mean(dim=1, keepdim=True)
    lap_x = laplacian_filter(x)
    lap_r = laplacian_filter(recon)
    hf_map = (lap_x - lap_r).abs().mean(dim=1, keepdim=True)
    return raw_map, hf_map


def sample_valid_mask(mask_type, mask_ratio, device):
    for _ in range(config.MAX_MASK_SAMPLE_TRIES):
        mask, actual_ratio = sample_mask(config.GRID_SIZE, mask_ratio, mask_type)
        mask = mask.to(device)
        pixel_mask = patch_mask_to_pixel_mask(mask, config.IMG_SIZE, config.PATCH_SIZE)
        core_mask = erode_pixel_mask(pixel_mask.unsqueeze(0), config.CORE_EROSION_PX)[0]
        if int(core_mask.sum().item()) >= config.CORE_MIN_PIXELS:
            return mask, actual_ratio, pixel_mask, core_mask
    raise RuntimeError(f"无法采到有效 mask: {mask_type}, {mask_ratio}")


@torch.no_grad()
def collect_k_predictions(model, x, mask_type, mask_ratio, K):
    recons = []
    pred_fulls = []
    raw_scores = []
    hf_scores = []
    core_masks = []
    pixel_masks = []
    ratios = []
    for _ in range(K):
        mask, actual_ratio, pixel_mask, core_mask = sample_valid_mask(mask_type, mask_ratio, x.device)
        recon, pred_full, _, _ = model.reconstruct(x, mask.unsqueeze(0), copy_back=True)
        raw_map, hf_map = compute_error_maps(x, recon)
        core = core_mask.unsqueeze(0)
        raw = (raw_map * core).sum() / core.sum().clamp_min(1.0)
        hf = (hf_map * core).sum() / core.sum().clamp_min(1.0)
        recons.append(recon)
        pred_fulls.append(pred_full)
        raw_scores.append(raw.item())
        hf_scores.append(hf.item())
        core_masks.append(core)
        pixel_masks.append(pixel_mask.unsqueeze(0))
        ratios.append(actual_ratio)
    return {
        "recons": recons,
        "pred_fulls": pred_fulls,
        "raw_scores": raw_scores,
        "hf_scores": hf_scores,
        "core_masks": core_masks,
        "pixel_masks": pixel_masks,
        "actual_ratios": ratios,
    }


@torch.no_grad()
def compute_var_heatmap(recons, core_masks):
    x_stack = torch.stack([r[0] for r in recons], dim=0)
    m_stack = torch.stack([m[0] for m in core_masks], dim=0).float()
    m_stack = m_stack.expand(-1, x_stack.shape[1], -1, -1)
    count = m_stack.sum(dim=0)
    valid = count >= 2.0
    sum_x = (x_stack * m_stack).sum(dim=0)
    sum_x2 = (x_stack.pow(2) * m_stack).sum(dim=0)
    mean = sum_x / count.clamp_min(1.0)
    var = sum_x2 / count.clamp_min(1.0) - mean.pow(2)
    var = var.mean(dim=0, keepdim=True)
    valid2d = valid.any(dim=0, keepdim=True)
    var = var * valid2d.float()
    return var


def get_dataset(split, return_path=False):
    if split == "test":
        return ChameleonTestDataset(return_path=return_path)
    if split == "val":
        class _ValDataset(RealPatchDataset):
            def __getitem__(self, idx):
                img = super().__getitem__(idx)
                if return_path:
                    path = self.paths[idx // self.patches_per_image]
                    return img, 0, path
                return img, 0
        return _ValDataset(split="val", patches_per_image=1)
    raise ValueError(f"未知 split: {split}")


def visualize(args):
    set_seed(config.SEED)
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    out_dir = os.path.join(exp_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_path, epoch = load_model(exp_dir, device)
    print(f"加载模型: {ckpt_path} (epoch={epoch})")

    dataset = get_dataset(args.split, return_path=True)
    total = min(args.num_samples, len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:total]

    for vis_idx, ds_idx in enumerate(indices):
        x, label, path = dataset[ds_idx]
        x = x.unsqueeze(0).to(device)
        out = collect_k_predictions(model, x, args.mask_type, args.mask_ratio, args.K)

        recon = out["recons"][0]
        pred_full = out["pred_fulls"][0]
        pixel_mask = out["pixel_masks"][0]
        core_mask = out["core_masks"][0]
        raw_map, hf_map = compute_error_maps(x, recon)
        var_map = compute_var_heatmap(out["recons"], out["core_masks"])

        raw_mean = float(np.mean(out["raw_scores"]))
        hf_mean = float(np.mean(out["hf_scores"]))
        ratio_mean = float(np.mean(out["actual_ratios"]))

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(
            f"{os.path.basename(path)} | label={label} | {args.mask_type}_{int(args.mask_ratio * 100)}% | "
            f"K={args.K} | raw={raw_mean:.4f} | hf={hf_mean:.4f} | ratio={ratio_mean:.4f}",
            fontsize=11,
        )

        axes[0, 0].imshow(tensor_to_rgb(x[0]))
        axes[0, 0].set_title("Input")
        axes[0, 1].imshow(tensor_to_rgb(pred_full[0]))
        axes[0, 1].set_title("Decoder Pred (full)")
        axes[0, 2].imshow(tensor_to_rgb(recon[0]))
        axes[0, 2].set_title("Visible Copy-back")
        axes[0, 3].imshow(pixel_mask[0, 0].detach().cpu().numpy(), cmap="gray")
        axes[0, 3].set_title("Pixel Mask")

        raw_arr, raw_min, raw_max = to_heatmap(raw_map[0, 0])
        hf_arr, hf_min, hf_max = to_heatmap(hf_map[0, 0])
        var_arr, var_min, var_max = to_heatmap(var_map[0, 0])
        core_arr = core_mask[0, 0].detach().cpu().numpy()

        im1 = axes[1, 0].imshow(core_arr, cmap="gray")
        axes[1, 0].set_title("Masked Core")
        im2 = axes[1, 1].imshow(raw_arr, cmap="magma", vmin=raw_min, vmax=raw_max)
        axes[1, 1].set_title("Raw Error Map")
        im3 = axes[1, 2].imshow(hf_arr, cmap="magma", vmin=hf_min, vmax=hf_max)
        axes[1, 2].set_title("High-freq Error Map")
        im4 = axes[1, 3].imshow(var_arr, cmap="viridis", vmin=var_min, vmax=var_max)
        axes[1, 3].set_title("K-run Pixel Var")

        for ax in axes.flat:
            ax.axis("off")

        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
        fig.tight_layout()

        save_name = f"vis_{vis_idx:03d}_{os.path.splitext(os.path.basename(path))[0]}.png"
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验三重建与误差可视化")
    parser.add_argument("--exp_name", type=str, default="Exp3_A0")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--mask_type", type=str, default="block", choices=["random", "block", "stripe"])
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()
    visualize(args)
