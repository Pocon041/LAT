import os
import torch
import torch.nn as nn
import lpips


VGG_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")


class FreqLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_recon):
        fft_x = torch.fft.rfft2(x, norm="ortho")
        fft_r = torch.fft.rfft2(x_recon, norm="ortho")
        return (fft_x.abs() - fft_r.abs()).abs().mean()


def load_lpips_vgg(device):
    fn = lpips.LPIPS(net="vgg").to(device)
    fn.eval()
    for p in fn.parameters():
        p.requires_grad = False
    return fn


class LightLoss(nn.Module):
    """L1 + Freq only, 无 LPIPS, 用于 masked 路径加速"""
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


class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_lpips=0.5, lambda_freq=0.1, device="cuda"):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_freq = lambda_freq
        self.l1 = nn.L1Loss()
        self.freq = FreqLoss()
        self.lpips_fn = load_lpips_vgg(device)

    def forward(self, x, x_recon):
        loss_l1 = self.l1(x_recon, x)
        loss_lpips = self.lpips_fn(x * 2 - 1, x_recon * 2 - 1).mean()
        loss_freq = self.freq(x, x_recon)
        total = (self.lambda_l1 * loss_l1
                 + self.lambda_lpips * loss_lpips
                 + self.lambda_freq * loss_freq)
        return total, {
            "l1": loss_l1.item(),
            "lpips": loss_lpips.item(),
            "freq": loss_freq.item(),
            "total": total.item(),
        }
