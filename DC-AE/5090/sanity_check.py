"""
Sanity Check: 辅助头交叉重建验证

验证文档中的假设:
1. StructHead (前32通道) 重建 X_low (低频结构)
2. DetailHead (后96通道) 重建 X_detail = X - X_low (细节残差)
3. 交叉重建应更差:
   - StructHead(Z_d) 重建 X_low 应比 StructHead(Z_s) 差
   - DetailHead(Z_s) 重建 X_detail 应比 DetailHead(Z_d) 差
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia
import matplotlib.pyplot as plt

import config
from dataset import RealPatchDataset, ChameleonTestDataset
from model import LatentMAE


def get_gaussian_blur(kernel_size, sigma):
    return kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size), (sigma, sigma)
    )


def compute_psnr(x, y):
    """计算 PSNR (dB)"""
    mse = F.mse_loss(x, y).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(x, y):
    """简化 SSIM (单通道)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return ssim.item()


@torch.no_grad()
def sanity_check(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    ae_path = os.path.join(exp_dir, "best_ae.pth")
    
    if not os.path.exists(ae_path):
        print(f"错误: 找不到 AE 权重: {ae_path}")
        return
    
    # 加载模型
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)
    ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.decoder.load_state_dict(ckpt["decoder"])
    if "struct_head" in ckpt:
        model.struct_head.load_state_dict(ckpt["struct_head"])
    if "detail_head" in ckpt:
        model.detail_head.load_state_dict(ckpt["detail_head"])
    model.eval()
    print(f"加载 AE: {ae_path} (epoch={ckpt['epoch']})")
    
    # 高斯模糊
    blur = get_gaussian_blur(
        config.S1_BLUR_KERNEL_SIZE, config.S1_BLUR_SIGMA
    ).to(device)
    
    struct_ch = config.STRUCT_CHANNELS
    
    # 数据
    ds_val = RealPatchDataset(split="val", preload=True, num_threads=8)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False)
    
    # 统计
    results = {
        "正常重建": {"struct_psnr": [], "detail_psnr": [], "full_psnr": []},
        "交叉重建": {"struct_from_d_psnr": [], "detail_from_s_psnr": []},
    }
    
    print("\n运行 Sanity Check...")
    for i, x in enumerate(tqdm(dl_val, desc="验证集")):
        if i >= args.num_samples:
            break
        
        x = x.to(device)
        x_low = blur(x)
        x_detail = x - x_low
        
        # 编码
        z = model.encoder(x)  # [1, 128, 16, 16]
        z_s = z[:, :struct_ch]   # [1, 32, 16, 16]
        z_d = z[:, struct_ch:]   # [1, 96, 16, 16]
        
        # 正常重建
        x_recon = model.decoder(z)
        x_low_recon = model.struct_head(z_s)
        x_detail_recon = model.detail_head(z_d)
        
        # 交叉重建 (错误的通道组合)
        x_low_from_d = model.struct_head(z_d[:, :struct_ch])  # 用细节通道前32去重建结构
        x_detail_from_s = model.detail_head(
            F.pad(z_s, (0, 0, 0, 0, 0, 128 - struct_ch - struct_ch))[:, :128-struct_ch]
        ) if struct_ch <= 64 else None
        
        # 计算指标
        results["正常重建"]["struct_psnr"].append(compute_psnr(x_low_recon, x_low))
        results["正常重建"]["detail_psnr"].append(compute_psnr(x_detail_recon, x_detail))
        results["正常重建"]["full_psnr"].append(compute_psnr(x_recon, x))
        
        results["交叉重建"]["struct_from_d_psnr"].append(compute_psnr(x_low_from_d, x_low))
        if x_detail_from_s is not None:
            results["交叉重建"]["detail_from_s_psnr"].append(
                compute_psnr(x_detail_from_s, x_detail)
            )
    
    # 汇总
    print("\n" + "=" * 60)
    print("Sanity Check 结果")
    print("=" * 60)
    
    print("\n[正常重建] (期望: 高 PSNR)")
    print(f"  Full recon PSNR:   {np.mean(results['正常重建']['full_psnr']):.2f} dB")
    print(f"  X_low (struct):    {np.mean(results['正常重建']['struct_psnr']):.2f} dB")
    print(f"  X_detail (detail): {np.mean(results['正常重建']['detail_psnr']):.2f} dB")
    
    print("\n[交叉重建] (期望: 低 PSNR, 比正常重建差)")
    print(f"  X_low from Z_d:    {np.mean(results['交叉重建']['struct_from_d_psnr']):.2f} dB")
    if results["交叉重建"]["detail_from_s_psnr"]:
        print(f"  X_detail from Z_s: {np.mean(results['交叉重建']['detail_from_s_psnr']):.2f} dB")
    
    # 验证假设
    struct_normal = np.mean(results['正常重建']['struct_psnr'])
    struct_cross = np.mean(results['交叉重建']['struct_from_d_psnr'])
    detail_normal = np.mean(results['正常重建']['detail_psnr'])
    
    print("\n[假设验证]")
    if struct_normal > struct_cross + 1.0:
        print(f"  StructHead: 正常 > 交叉 ({struct_normal:.2f} > {struct_cross:.2f}) -> PASS")
    else:
        print(f"  StructHead: 正常 vs 交叉 ({struct_normal:.2f} vs {struct_cross:.2f}) -> FAIL (差距不明显)")
    
    # 可视化
    if args.visualize:
        save_dir = os.path.join(exp_dir, "sanity_check")
        os.makedirs(save_dir, exist_ok=True)
        
        # 取最后一个样本可视化 (确保在同一设备上)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        def to_np(t):
            return t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        
        def to_np_detail(t):
            # 细节残差放大显示
            arr = t[0].permute(1, 2, 0).cpu().numpy()
            return (arr * 2 + 0.5).clip(0, 1)
        
        # 预计算 PSNR (确保同设备)
        psnr_full = compute_psnr(x_recon.cpu(), x.cpu())
        psnr_struct = compute_psnr(x_low_recon.cpu(), x_low.cpu())
        psnr_detail = compute_psnr(x_detail_recon.cpu(), x_detail.cpu())
        psnr_cross = compute_psnr(x_low_from_d.cpu(), x_low.cpu())
        
        axes[0, 0].imshow(to_np(x))
        axes[0, 0].set_title("Input X")
        axes[0, 1].imshow(to_np(x_low))
        axes[0, 1].set_title("X_low (blur)")
        axes[0, 2].imshow(to_np_detail(x_detail))
        axes[0, 2].set_title("X_detail (residual)")
        axes[0, 3].imshow(to_np(x_recon))
        axes[0, 3].set_title(f"Full Recon\nPSNR={psnr_full:.1f}")
        
        axes[1, 0].imshow(to_np(x_low_recon))
        axes[1, 0].set_title(f"StructHead(Z_s)\nPSNR={psnr_struct:.1f}")
        axes[1, 1].imshow(to_np_detail(x_detail_recon))
        axes[1, 1].set_title(f"DetailHead(Z_d)\nPSNR={psnr_detail:.1f}")
        axes[1, 2].imshow(to_np(x_low_from_d))
        axes[1, 2].set_title(f"StructHead(Z_d) [CROSS]\nPSNR={psnr_cross:.1f}")
        axes[1, 3].axis("off")
        
        for ax in axes.flat:
            ax.axis("off")
        
        plt.tight_layout()
        fig_path = os.path.join(save_dir, "sanity_check.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n可视化保存到: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="辅助头 Sanity Check")
    parser.add_argument("--exp_name", type=str, default="A0", help="实验名称")
    parser.add_argument("--num_samples", type=int, default=50, help="验证样本数")
    parser.add_argument("--visualize", action="store_true", help="是否保存可视化")
    args = parser.parse_args()
    sanity_check(args)
