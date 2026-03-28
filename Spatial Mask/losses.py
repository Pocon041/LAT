import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqLoss(nn.Module):
    """频域 L1 损失: 比较 FFT 幅度谱"""
    def __init__(self):
        super().__init__()

    def forward(self, x, x_recon):
        fft_x = torch.fft.rfft2(x, norm="ortho")
        fft_r = torch.fft.rfft2(x_recon, norm="ortho")
        return (fft_x.abs() - fft_r.abs()).abs().mean()


class Stage1Loss(nn.Module):
    """
    阶段一损失: L1 + 频域损失
    L_stage1 = lambda_l1 * L1 + lambda_freq * FreqLoss
    """
    def __init__(self, lambda_l1=1.0, lambda_freq=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.l1 = nn.L1Loss()
        self.freq = FreqLoss()

    def forward(self, x, x_recon):
        loss_l1 = self.l1(x_recon, x)
        loss_freq = self.freq(x, x_recon)
        total = self.lambda_l1 * loss_l1 + self.lambda_freq * loss_freq
        return total, {
            "l1": loss_l1.item(),
            "freq": loss_freq.item(),
            "total": total.item(),
        }


class Stage2Loss(nn.Module):
    """
    阶段二损失: 只在被 mask 的 token 上计算
    L_stage2 = lambda_l1 * MaskedLatentL1 + lambda_cos * MaskedLatentCos
    """
    def __init__(self, lambda_l1=1.0, lambda_cos=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cos = lambda_cos

    def forward(self, z, z_hat, mask):
        """
        z: [B, C, H, W] 原始 latent
        z_hat: [B, C, H, W] 补全后的 latent
        mask: [B, N] bool, True = 被 mask
        """
        B, C, H, W = z.shape

        # 展平为 [B, N, C]
        z_flat = z.flatten(2).permute(0, 2, 1)
        z_hat_flat = z_hat.flatten(2).permute(0, 2, 1)

        # 只取被 mask 的 token
        # mask: [B, N] -> 收集所有被 mask 的 token
        mask_f = mask.float()  # [B, N]
        num_masked = mask_f.sum()

        if num_masked == 0:
            zero = torch.tensor(0.0, device=z.device)
            return zero, {"l1": 0.0, "cos": 0.0, "total": 0.0}

        # Masked L1
        diff = (z_flat - z_hat_flat).abs()  # [B, N, C]
        masked_l1 = (diff * mask_f.unsqueeze(-1)).sum() / (num_masked * C)

        # Masked Cosine: 1 - cos_sim (越小越好)
        z_masked = z_flat[mask]       # [M, C]
        z_hat_masked = z_hat_flat[mask]  # [M, C]
        cos_sim = F.cosine_similarity(z_masked, z_hat_masked, dim=-1)  # [M]
        masked_cos = (1.0 - cos_sim).mean()

        total = self.lambda_l1 * masked_l1 + self.lambda_cos * masked_cos
        return total, {
            "l1": masked_l1.item(),
            "cos": masked_cos.item(),
            "total": total.item(),
        }
