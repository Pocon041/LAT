import torch
import torch.nn as nn
import torch.nn.functional as F

import config


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
    阶段一损失 (Structure-Aware):
    L_stage1 = L_full-recon + λ_s * L_struct + λ_d * L_detail
    
    其中:
    - L_full-recon = L_pixel-L1 + λ_f * L_freq
    - L_struct = α1*L1 + α2*L_cos (Z_s -> X_low)
    - L_detail = L1 (Z_d -> X_detail = X - X_low)
    """
    def __init__(
        self,
        lambda_l1=1.0,
        lambda_freq=0.1,
        lambda_struct=0.5,
        struct_l1_weight=1.0,
        struct_cos_weight=0.2,
        lambda_detail=0.1,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_freq = lambda_freq
        self.lambda_struct = lambda_struct
        self.struct_l1_weight = struct_l1_weight
        self.struct_cos_weight = struct_cos_weight
        self.lambda_detail = lambda_detail
        
        self.l1 = nn.L1Loss()
        self.freq = FreqLoss()

    def forward(self, x, x_recon, x_low, x_struct_pred, x_detail_target, x_detail_pred):
        """
        x: 原图
        x_recon: 主解码器重建
        x_low: 高斯模糊后的低频目标
        x_struct_pred: H_s(Z_s) 预测的低频图
        x_detail_target: X - X_low 细节残差目标
        x_detail_pred: H_d(Z_d) 预测的细节残差
        """
        # L_full-recon
        loss_l1 = self.l1(x_recon, x)
        loss_freq = self.freq(x, x_recon)
        loss_full_recon = self.lambda_l1 * loss_l1 + self.lambda_freq * loss_freq
        
        # L_struct: 结构辅助头损失
        struct_l1 = self.l1(x_struct_pred, x_low)
        # cosine similarity (flatten to compare direction)
        x_low_flat = x_low.flatten(1)
        x_struct_flat = x_struct_pred.flatten(1)
        cos_sim = F.cosine_similarity(x_low_flat, x_struct_flat, dim=-1).mean()
        struct_cos = 1.0 - cos_sim
        loss_struct = self.struct_l1_weight * struct_l1 + self.struct_cos_weight * struct_cos
        
        # L_detail: 细节辅助头损失
        loss_detail = self.l1(x_detail_pred, x_detail_target)
        
        # 总损失
        total = loss_full_recon + self.lambda_struct * loss_struct + self.lambda_detail * loss_detail
        
        return total, {
            "l1": loss_l1.item(),
            "freq": loss_freq.item(),
            "full_recon": loss_full_recon.item(),
            "struct_l1": struct_l1.item(),
            "struct_cos": struct_cos.item(),
            "struct": loss_struct.item(),
            "detail": loss_detail.item(),
            "total": total.item(),
        }


class Stage2Loss(nn.Module):
    """
    阶段二损失 (Structure-Aware, 分通道):
    L_stage2 = λ_s * L_mask-struct + λ_d * L_mask-detail
    
    其中:
    - L_mask-struct = α1*L1^(s) + α2*L_cos^(s)  只在前32通道
    - L_mask-detail = β1*L1^(d) + β2*L_cos^(d)  只在后96通道
    """
    def __init__(
        self,
        struct_channels=32,
        lambda_struct=1.0,
        struct_l1_weight=1.0,
        struct_cos_weight=0.3,
        lambda_detail=0.08,
        detail_l1_weight=1.0,
        detail_cos_weight=0.3,
    ):
        super().__init__()
        self.struct_channels = struct_channels
        self.lambda_struct = lambda_struct
        self.struct_l1_weight = struct_l1_weight
        self.struct_cos_weight = struct_cos_weight
        self.lambda_detail = lambda_detail
        self.detail_l1_weight = detail_l1_weight
        self.detail_cos_weight = detail_cos_weight

    def _masked_l1_cos(self, z, z_hat, mask, channel_slice):
        """
        计算指定通道范围内的 masked L1 和 cosine loss
        z, z_hat: [B, C, H, W]
        mask: [B, N] bool
        channel_slice: slice object, e.g., slice(0, 32)
        """
        B, C, H, W = z.shape
        
        # 取指定通道
        z_ch = z[:, channel_slice]        # [B, C_sub, H, W]
        z_hat_ch = z_hat[:, channel_slice]
        C_sub = z_ch.shape[1]
        
        # 展平为 [B, N, C_sub]
        z_flat = z_ch.flatten(2).permute(0, 2, 1)
        z_hat_flat = z_hat_ch.flatten(2).permute(0, 2, 1)
        
        mask_f = mask.float()  # [B, N]
        num_masked = mask_f.sum()
        
        if num_masked == 0:
            zero = torch.tensor(0.0, device=z.device)
            return zero, zero
        
        # Masked L1
        diff = (z_flat - z_hat_flat).abs()  # [B, N, C_sub]
        masked_l1 = (diff * mask_f.unsqueeze(-1)).sum() / (num_masked * C_sub)
        
        # Masked Cosine
        z_masked = z_flat[mask]       # [M, C_sub]
        z_hat_masked = z_hat_flat[mask]
        cos_sim = F.cosine_similarity(z_masked, z_hat_masked, dim=-1)
        masked_cos = (1.0 - cos_sim).mean()
        
        return masked_l1, masked_cos

    def forward(self, z, z_hat, mask):
        """
        z: [B, C, H, W] 原始 latent
        z_hat: [B, C, H, W] 补全后的 latent
        mask: [B, N] bool, True = 被 mask
        """
        # 结构通道 (前32)
        struct_slice = slice(0, self.struct_channels)
        struct_l1, struct_cos = self._masked_l1_cos(z, z_hat, mask, struct_slice)
        loss_struct = self.struct_l1_weight * struct_l1 + self.struct_cos_weight * struct_cos
        
        # 细节通道 (后96)
        detail_slice = slice(self.struct_channels, None)
        detail_l1, detail_cos = self._masked_l1_cos(z, z_hat, mask, detail_slice)
        loss_detail = self.detail_l1_weight * detail_l1 + self.detail_cos_weight * detail_cos
        
        # 总损失
        total = self.lambda_struct * loss_struct + self.lambda_detail * loss_detail
        
        return total, {
            "struct_l1": struct_l1.item(),
            "struct_cos": struct_cos.item(),
            "struct": loss_struct.item(),
            "detail_l1": detail_l1.item(),
            "detail_cos": detail_cos.item(),
            "detail": loss_detail.item(),
            "total": total.item(),
        }
