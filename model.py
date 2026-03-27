import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch_list=None, latent_channels=128):
        super().__init__()
        if ch_list is None:
            ch_list = [64, 128, 256, 128]
        layers = []
        c_in = in_channels
        for c_out in ch_list:
            layers.append(nn.Conv2d(c_in, c_out, 4, 2, 1))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(ResBlock(c_out))
            c_in = c_out
        if ch_list[-1] != latent_channels:
            layers.append(nn.Conv2d(ch_list[-1], latent_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels=128, ch_list=None, out_channels=3):
        super().__init__()
        if ch_list is None:
            ch_list = [256, 128, 64]
        layers = []
        c_in = latent_channels
        for c_out in ch_list:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(c_in, c_out, 3, 1, 1))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(ResBlock(c_out))
            c_in = c_out
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(c_in, out_channels, 3, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def channel_mask(z, keep_ratio):
    """per-sample channel-level mask, 空间维共享"""
    B, C, H, W = z.shape
    n_keep = max(1, int(C * keep_ratio))
    z_masked = torch.zeros_like(z)
    for b in range(B):
        perm = torch.randperm(C, device=z.device)[:n_keep]
        z_masked[b, perm] = z[b, perm]
    return z_masked


class SimpleAE(nn.Module):
    """
    朴素卷积自编码器, 支持 Latent MAE 双路径训练
    输入: [B, 3, 256, 256]
    Latent: [B, 128, 16, 16]
    输出: [B, 3, 256, 256]
    """
    def __init__(self, latent_channels=128):
        super().__init__()
        self.encoder = Encoder(latent_channels=latent_channels)
        self.decoder = Decoder(latent_channels=latent_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, mask_keep_ratio=None):
        """
        mask_keep_ratio=None: 普通 full 路径
        mask_keep_ratio=float: 返回 full 重建和 masked 重建
        """
        z = self.encode(x)
        recon_full = self.decode(z)

        if mask_keep_ratio is not None:
            z_m = channel_mask(z, mask_keep_ratio)
            recon_mask = self.decode(z_m)
            return recon_full, recon_mask, z
        else:
            return recon_full, z


if __name__ == "__main__":
    model = SimpleAE(latent_channels=128)
    x = torch.randn(2, 3, 256, 256)
    recon_full, recon_mask, z = model(x, mask_keep_ratio=0.5)
    print(f"输入: {x.shape}")
    print(f"Latent: {z.shape}")
    print(f"Full 重建: {recon_full.shape}")
    print(f"Masked 重建: {recon_mask.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
