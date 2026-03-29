import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia

import config
from model import LatentMAE
from dataset import _collect_images, _resize_shorter_side


def load_ae(exp_dir, device):
    """加载 AE (Encoder + Decoder + StructHead + DetailHead)"""
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    ae_path = os.path.join(exp_dir, "best_ae.pth")
    ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if "struct_head" in ckpt:
        model.struct_head.load_state_dict(ckpt["struct_head"])
    if "detail_head" in ckpt:
        model.detail_head.load_state_dict(ckpt["detail_head"])
    print(f"加载 AE: {ae_path} (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    model.eval()
    return model


def get_gaussian_blur(kernel_size, sigma):
    return kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)
    )


def preprocess(img_path):
    """与训练验证集一致的确定性预处理"""
    img = Image.open(img_path).convert("RGB")
    img = _resize_shorter_side(img, target_short=288)
    transform = transforms.Compose([
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
    ])
    return transform(img)


@torch.no_grad()
def reconstruct(model, img_tensor, device, blur=None):
    """
    重构单张图像
    返回: x_recon, x_struct, x_detail, x_low, x_high
    """
    x = img_tensor.unsqueeze(0).to(device)
    x_recon, z, x_struct, x_detail = model.forward_stage1(x)
    
    # 生成 GT: x_low (blur) 和 x_high (residual)
    if blur is not None:
        x_low = blur(x)
        x_high = x - x_low
    else:
        x_low = x
        x_high = torch.zeros_like(x)
    
    return (
        x_recon[0].cpu().clamp(0, 1),
        x_struct[0].cpu().clamp(0, 1),
        x_detail[0].cpu(),  # 残差可能有负值
        x_low[0].cpu().clamp(0, 1),
        x_high[0].cpu(),
    )


def tensor_to_numpy(t):
    """[C, H, W] tensor -> [H, W, C] numpy for matplotlib"""
    return t.permute(1, 2, 0).numpy()


def compute_residual(orig, recon):
    """计算残差图 (放大显示)"""
    diff = (orig - recon).abs()
    # 放大 5 倍方便观察
    diff = (diff * 5).clamp(0, 1)
    return diff


def detail_to_vis(t):
    """将残差转换为可视化格式 (放大+偏移)"""
    arr = t.permute(1, 2, 0).numpy()
    return (arr * 3 + 0.5).clip(0, 1)


def visualize(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    out_dir = os.path.join(exp_dir, "vis_recon")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ae(exp_dir, device)
    
    # 高斯模糊 (与训练一致)
    blur = get_gaussian_blur(
        config.S1_BLUR_KERNEL_SIZE, config.S1_BLUR_SIGMA
    ).to(device)

    # 收集图像
    real_paths = _collect_images(config.CHAMELEON_REAL_DIR)
    fake_paths = _collect_images(config.CHAMELEON_FAKE_DIR)
    val_paths = _collect_images(config.DIV2K_VAL_DIR)

    # 随机采样
    random.seed(42)
    n = args.num_samples
    real_samples = random.sample(real_paths, min(n, len(real_paths)))
    fake_samples = random.sample(fake_paths, min(n, len(fake_paths)))
    val_samples = random.sample(val_paths, min(n, len(val_paths)))

    sources = [
        ("real_Chameleon", real_samples),
        ("fake_Chameleon", fake_samples),
        ("val_DIV2K", val_samples),
    ]

    # ========== 完整重建可视化 (含 struct/detail 头) ==========
    for src_name, paths in sources:
        num = len(paths)
        # 7列: Original, X_low(GT), X_high(GT), Full_Recon, Struct_Recon, Detail_Recon, Residual
        fig, axes = plt.subplots(num, 7, figsize=(28, 4 * num))
        if num == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"Struct/Detail Reconstruction: {src_name} ({args.exp_name})", fontsize=14, y=0.99)

        for i, p in enumerate(paths):
            orig = preprocess(p)
            x_recon, x_struct, x_detail, x_low, x_high = reconstruct(model, orig, device, blur)
            
            # L1 误差
            l1_full = (orig - x_recon).abs().mean().item()
            l1_struct = (x_low - x_struct).abs().mean().item()
            l1_detail = (x_high - x_detail).abs().mean().item()

            axes[i, 0].imshow(tensor_to_numpy(orig))
            axes[i, 0].set_title("Original", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(tensor_to_numpy(x_low))
            axes[i, 1].set_title("X_low (GT)", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(detail_to_vis(x_high))
            axes[i, 2].set_title("X_high (GT)", fontsize=9)
            axes[i, 2].axis("off")

            axes[i, 3].imshow(tensor_to_numpy(x_recon))
            axes[i, 3].set_title(f"Full Recon\nL1={l1_full:.4f}", fontsize=9)
            axes[i, 3].axis("off")

            axes[i, 4].imshow(tensor_to_numpy(x_struct))
            axes[i, 4].set_title(f"StructHead\nL1={l1_struct:.4f}", fontsize=9)
            axes[i, 4].axis("off")

            axes[i, 5].imshow(detail_to_vis(x_detail))
            axes[i, 5].set_title(f"DetailHead\nL1={l1_detail:.4f}", fontsize=9)
            axes[i, 5].axis("off")

            residual = compute_residual(orig, x_recon)
            axes[i, 6].imshow(tensor_to_numpy(residual))
            axes[i, 6].set_title("Residual (x5)", fontsize=9)
            axes[i, 6].axis("off")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"struct_detail_{src_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"保存: {save_path}")

    # ========== Real vs Fake 对比图 ==========
    num_compare = min(n, len(real_samples), len(fake_samples))
    fig, axes = plt.subplots(num_compare, 6, figsize=(24, 4 * num_compare))
    if num_compare == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Real vs Fake Comparison ({args.exp_name})", fontsize=14, y=0.99)

    for i in range(num_compare):
        # Real
        real_orig = preprocess(real_samples[i])
        r_recon, r_struct, r_detail, r_low, r_high = reconstruct(model, real_orig, device, blur)
        r_l1 = (real_orig - r_recon).abs().mean().item()

        # Fake
        fake_orig = preprocess(fake_samples[i])
        f_recon, f_struct, f_detail, f_low, f_high = reconstruct(model, fake_orig, device, blur)
        f_l1 = (fake_orig - f_recon).abs().mean().item()

        axes[i, 0].imshow(tensor_to_numpy(real_orig))
        axes[i, 0].set_title(f"Real Original", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tensor_to_numpy(r_recon))
        axes[i, 1].set_title(f"Real Recon\nL1={r_l1:.4f}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tensor_to_numpy(compute_residual(real_orig, r_recon)))
        axes[i, 2].set_title("Real Res (x5)", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(tensor_to_numpy(fake_orig))
        axes[i, 3].set_title(f"Fake Original", fontsize=9)
        axes[i, 3].axis("off")

        axes[i, 4].imshow(tensor_to_numpy(f_recon))
        axes[i, 4].set_title(f"Fake Recon\nL1={f_l1:.4f}", fontsize=9)
        axes[i, 4].axis("off")

        axes[i, 5].imshow(tensor_to_numpy(compute_residual(fake_orig, f_recon)))
        axes[i, 5].set_title("Fake Res (x5)", fontsize=9)
        axes[i, 5].axis("off")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "real_vs_fake.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {save_path}")

    print(f"\n所有可视化已保存到: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AE 重构可视化")
    parser.add_argument("--exp_name", type=str, default="Test1",
                        help="实验名称")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="每类采样数量")
    args = parser.parse_args()
    visualize(args)
