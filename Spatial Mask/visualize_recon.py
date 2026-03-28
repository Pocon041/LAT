import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import config
from model import LatentMAE
from dataset import _collect_images, _resize_shorter_side


def load_ae(exp_dir, device):
    """只加载 AE (Encoder + Decoder)"""
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    ae_path = os.path.join(exp_dir, "best_ae.pth")
    ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    print(f"加载 AE: {ae_path} (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    model.eval()
    return model


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
def reconstruct(model, img_tensor, device):
    """重构单张图像, 返回重构后的 tensor"""
    x = img_tensor.unsqueeze(0).to(device)
    x_recon, z = model.forward_stage1(x)
    x_recon = x_recon.clamp(0, 1)
    return x_recon[0].cpu()


def tensor_to_numpy(t):
    """[C, H, W] tensor -> [H, W, C] numpy for matplotlib"""
    return t.permute(1, 2, 0).numpy()


def compute_residual(orig, recon):
    """计算残差图 (放大显示)"""
    diff = (orig - recon).abs()
    # 放大 5 倍方便观察
    diff = (diff * 5).clamp(0, 1)
    return diff


def visualize(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    out_dir = os.path.join(exp_dir, "vis_recon")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ae(exp_dir, device)

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
        ("real (Chameleon)", real_samples),
        ("fake (Chameleon)", fake_samples),
        ("val (DIV2K)", val_samples),
    ]

    for src_name, paths in sources:
        num = len(paths)
        fig, axes = plt.subplots(num, 3, figsize=(12, 4 * num))
        if num == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f"AE Reconstruction: {src_name} ({args.exp_name})", fontsize=14, y=0.99)

        for i, p in enumerate(paths):
            orig = preprocess(p)
            recon = reconstruct(model, orig, device)
            residual = compute_residual(orig, recon)

            l1_err = (orig - recon).abs().mean().item()

            axes[i, 0].imshow(tensor_to_numpy(orig))
            axes[i, 0].set_title("Original", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(tensor_to_numpy(recon))
            axes[i, 1].set_title(f"Recon (L1={l1_err:.4f})", fontsize=10)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(tensor_to_numpy(residual))
            axes[i, 2].set_title("Residual (x5)", fontsize=10)
            axes[i, 2].axis("off")

        plt.tight_layout()
        safe_name = src_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(out_dir, f"recon_{safe_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"保存: {save_path}")

    # 额外: 真假对比图 (同一张图里左 real 右 fake)
    num_compare = min(n, len(real_samples), len(fake_samples))
    fig, axes = plt.subplots(num_compare, 4, figsize=(16, 4 * num_compare))
    if num_compare == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Real vs Fake Reconstruction Comparison ({args.exp_name})", fontsize=14, y=0.99)

    for i in range(num_compare):
        real_orig = preprocess(real_samples[i])
        real_recon = reconstruct(model, real_orig, device)
        real_res = compute_residual(real_orig, real_recon)
        real_l1 = (real_orig - real_recon).abs().mean().item()

        fake_orig = preprocess(fake_samples[i])
        fake_recon = reconstruct(model, fake_orig, device)
        fake_res = compute_residual(fake_orig, fake_recon)
        fake_l1 = (fake_orig - fake_recon).abs().mean().item()

        axes[i, 0].imshow(tensor_to_numpy(real_orig))
        axes[i, 0].set_title(f"Real (L1={real_l1:.4f})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tensor_to_numpy(real_res))
        axes[i, 1].set_title("Real Residual (x5)", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tensor_to_numpy(fake_orig))
        axes[i, 2].set_title(f"Fake (L1={fake_l1:.4f})", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(tensor_to_numpy(fake_res))
        axes[i, 3].set_title("Fake Residual (x5)", fontsize=10)
        axes[i, 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "recon_real_vs_fake.png")
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
