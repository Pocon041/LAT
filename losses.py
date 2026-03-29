import torch
import torch.nn as nn
import torch.nn.functional as F


def build_laplacian_kernel(device, dtype, channels):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return kernel


def laplacian_filter(x):
    c = x.shape[1]
    kernel = build_laplacian_kernel(x.device, x.dtype, c)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x_pad, kernel, groups=c)


def masked_mean(x, mask, eps=1e-8):
    if mask.dtype != x.dtype:
        mask = mask.to(x.dtype)
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def masked_l1_mean(x_pred, x_true, pixel_mask, eps=1e-8):
    if pixel_mask.dim() == 3:
        pixel_mask = pixel_mask.unsqueeze(1)
    diff = (x_pred - x_true).abs()
    mask = pixel_mask.expand_as(diff)
    return masked_mean(diff, mask, eps=eps)


def masked_laplacian_l1_mean(x_pred, x_true, pixel_mask, eps=1e-8):
    if pixel_mask.dim() == 3:
        pixel_mask = pixel_mask.unsqueeze(1)
    lap_pred = laplacian_filter(x_pred)
    lap_true = laplacian_filter(x_true)
    diff = (lap_pred - lap_true).abs()
    mask = pixel_mask.expand_as(diff)
    return masked_mean(diff, mask, eps=eps)


class PixelReconstructionLoss(nn.Module):
    def __init__(self, lambda_raw=1.0, lambda_lap=0.5):
        super().__init__()
        self.lambda_raw = lambda_raw
        self.lambda_lap = lambda_lap

    def forward(self, x_true, x_pred, pixel_mask):
        raw = masked_l1_mean(x_pred, x_true, pixel_mask)
        lap = masked_laplacian_l1_mean(x_pred, x_true, pixel_mask)
        total = self.lambda_raw * raw + self.lambda_lap * lap
        return total, {
            "raw": raw.item(),
            "lap": lap.item(),
            "total": total.item(),
            "masked_pixels": float(pixel_mask.sum().item()),
        }
