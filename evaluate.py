import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
from dataset import ChameleonTestDataset, RealPatchDataset
from model import LatentMAE, sample_mask


def log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def load_model(exp_dir, device):
    """加载完整模型 (AE + Predictor)"""
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)

    ae_path = os.path.join(exp_dir, "best_ae.pth")
    pred_path = os.path.join(exp_dir, "best_predictor.pth")

    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ae_ckpt["encoder"])
    model.decoder.load_state_dict(ae_ckpt["decoder"])
    print(f"加载 AE: {ae_path} (epoch={ae_ckpt['epoch']})")

    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=True)
    model.predictor.load_state_dict(pred_ckpt["predictor"])
    print(f"加载 Predictor: {pred_path} (epoch={pred_ckpt['epoch']})")

    model.eval()
    return model


@torch.no_grad()
def compute_masked_error(model, z, mask_ratio, mask_type, K, grid_size):
    """
    对单个 latent z 做 K 次不同 mask 的补全, 计算平均误差
    z: [1, C, H, W]
    返回: mean_l1_err, mean_cos_err, per_run_z_hats, per_run_masks, mean_actual_ratio
    """
    B, C, H, W = z.shape
    N = H * W
    z_flat = z.flatten(2).permute(0, 2, 1)  # [1, N, C]

    l1_errors = []
    cos_errors = []
    z_hats_flat = []
    used_masks = []
    actual_ratios = []

    for _ in range(K):
        mask, actual_ratio = sample_mask(grid_size, mask_ratio, mask_type)
        actual_ratios.append(actual_ratio)
        mask = mask.unsqueeze(0).to(z.device)
        z_hat = model.predict(z, mask)
        z_hat_flat = z_hat.flatten(2).permute(0, 2, 1)  # [1, N, C]

        m = mask[0]  # [N]
        if m.sum() == 0:
            continue

        z_masked = z_flat[0, m]        # [M, C]
        z_hat_masked = z_hat_flat[0, m]  # [M, C]

        l1_err = (z_masked - z_hat_masked).abs().mean().item()
        l1_errors.append(l1_err)

        cos_sim = torch.nn.functional.cosine_similarity(
            z_masked, z_hat_masked, dim=-1
        ).mean().item()
        cos_errors.append(1.0 - cos_sim)

        z_hats_flat.append(z_hat_flat)
        used_masks.append(m)  # 保留实际使用的 mask

    mean_l1 = np.mean(l1_errors) if l1_errors else 0.0
    mean_cos = np.mean(cos_errors) if cos_errors else 0.0
    mean_actual_ratio = np.mean(actual_ratios) if actual_ratios else mask_ratio
    return mean_l1, mean_cos, z_hats_flat, used_masks, mean_actual_ratio


@torch.no_grad()
def compute_base_error(model, z, grid_size, mask_type="random",
                      base_ratio=None, base_runs=None):
    """
    计算 base 误差 (低 mask ratio, 作为归一化基准)
    mask_type 与评估条件保持一致, 避免混入 mask 形状差异
    """
    if base_ratio is None:
        base_ratio = config.EVAL_BASE_MASK_RATIO
    if base_runs is None:
        base_runs = config.EVAL_BASE_RUNS

    l1_errors = []
    cos_errors = []
    B, C, H, W = z.shape
    z_flat = z.flatten(2).permute(0, 2, 1)

    for _ in range(base_runs):
        mask, _ = sample_mask(grid_size, base_ratio, mask_type)
        mask = mask.unsqueeze(0).to(z.device)
        z_hat = model.predict(z, mask)
        z_hat_flat = z_hat.flatten(2).permute(0, 2, 1)

        m = mask[0]
        if m.sum() == 0:
            continue

        z_masked = z_flat[0, m]
        z_hat_masked = z_hat_flat[0, m]

        l1_err = (z_masked - z_hat_masked).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            z_masked, z_hat_masked, dim=-1
        ).mean().item()

        l1_errors.append(l1_err)
        cos_errors.append(1.0 - cos_sim)

    return np.mean(l1_errors), np.mean(cos_errors)


@torch.no_grad()
def compute_s_var(z_hats_flat_list, masks_list, grid_size):
    """
    计算 S_var: 对每个 token, 只统计该 token 被 mask 时的预测方差
    z_hats_flat_list: list of [1, N, C] tensors
    masks_list: list of [N] bool tensors
    返回标量方差
    """
    if len(z_hats_flat_list) < 2:
        return 0.0

    N = grid_size * grid_size
    C = z_hats_flat_list[0].shape[-1]
    device = z_hats_flat_list[0].device

    token_vars = []
    for i in range(N):
        # 收集该 token 在所有被 mask 的 run 中的预测
        preds = []
        for k, (z_hat, mask) in enumerate(zip(z_hats_flat_list, masks_list)):
            if mask[i]:
                preds.append(z_hat[0, i])  # [C]
        if len(preds) >= 2:
            preds_stack = torch.stack(preds)  # [K', C]
            var = preds_stack.var(dim=0).mean().item()
            token_vars.append(var)

    return np.mean(token_vars) if token_vars else 0.0


@torch.no_grad()
def compute_complexity_features(img_tensor):
    """
    计算图像复杂度特征 (用于相关性分析)
    img_tensor: [1, 3, H, W], 值域 [0, 1]
    返回 dict
    """
    x = img_tensor[0]  # [3, H, W]
    gray = x.mean(dim=0)  # [H, W]

    # 边缘密度 (Sobel)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = torch.nn.functional.conv2d(g, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(g, sobel_y, padding=1)
    edge_density = (gx ** 2 + gy ** 2).sqrt().mean().item()

    # 高频能量
    fft = torch.fft.rfft2(gray, norm="ortho")
    mag = fft.abs()
    H, W2 = mag.shape
    # 高频: 距离中心 > 0.5 * max_radius 的部分
    fy = torch.arange(H, device=x.device).float()
    fx = torch.arange(W2, device=x.device).float()
    fy[fy > H // 2] -= H
    gy_freq, gx_freq = torch.meshgrid(fy, fx, indexing="ij")
    radius = (gy_freq ** 2 + gx_freq ** 2).sqrt()
    max_r = radius.max()
    hf_mask = radius > 0.5 * max_r
    hf_energy = mag[hf_mask].pow(2).sum().item() / mag.pow(2).sum().item()

    # Patch entropy (将图像分成 8x8 patch, 计算灰度直方图熵的均值)
    patch_size = 32
    entropies = []
    for i in range(0, gray.shape[0] - patch_size + 1, patch_size):
        for j in range(0, gray.shape[1] - patch_size + 1, patch_size):
            patch = gray[i:i + patch_size, j:j + patch_size].flatten()
            hist = torch.histc(patch, bins=32, min=0.0, max=1.0)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            ent = -(hist * hist.log()).sum().item()
            entropies.append(ent)
    patch_entropy = np.mean(entropies) if entropies else 0.0

    # 梯度能量
    grad_energy = (gx ** 2 + gy ** 2).mean().item()

    return {
        "edge_density": edge_density,
        "hf_energy": hf_energy,
        "patch_entropy": patch_entropy,
        "grad_energy": grad_energy,
    }


@torch.no_grad()
def evaluate(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, "eval_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}", log_file)

    # 加载模型
    model = load_model(exp_dir, device)
    grid_size = config.LATENT_SPATIAL
    K = args.K

    # 加载测试集
    ds_test = ChameleonTestDataset()
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2)
    log(f"Chameleon 测试集: {len(ds_test)} 张", log_file)

    # --- 第一步: 在真实验证集上按 mask_type 分别统计 tau ---
    mask_types = ["random", "block"]
    eval_ratios = config.EVAL_MASK_RATIOS

    log("计算 tau (基于验证集 base 误差, 按 mask_type 分别统计)...", log_file)
    ds_val = RealPatchDataset(patches_per_image=1, split="val")
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=2)

    # 先提取所有验证集 latent
    val_latents = []
    for batch in tqdm(dl_val, desc="提取验证集 latent"):
        x = batch.to(device)
        z = model.encode(x)
        val_latents.append(z)

    tau_dict = {}  # {mask_type: {"l1": float, "cos": float}}
    for mtype in mask_types:
        base_l1_list = []
        base_cos_list = []
        for z in tqdm(val_latents, desc=f"  tau[{mtype}]", leave=False):
            bl1, bcos = compute_base_error(model, z, grid_size, mask_type=mtype)
            base_l1_list.append(bl1)
            base_cos_list.append(bcos)
        t_l1 = np.percentile(base_l1_list, config.TAU_PERCENTILE)
        t_cos = np.percentile(base_cos_list, config.TAU_PERCENTILE)
        tau_dict[mtype] = {"l1": t_l1, "cos": t_cos}
        log(f"  tau[{mtype}]: l1={t_l1:.6f}, cos={t_cos:.6f} "
            f"({config.TAU_PERCENTILE}% 分位数)", log_file)

    # --- 第二步: 对每个评估条件计算分数 ---
    all_results = {}

    for mtype in mask_types:
        for mratio in eval_ratios:
            condition = f"{mtype}_{int(mratio * 100)}%"
            log(f"\n--- 评估条件: {condition}, K={K} ---", log_file)

            scores_logratio_l1 = []
            scores_logratio_cos = []
            scores_var = []
            labels = []
            complexity_feats = []
            actual_ratios_all = []

            for x, label in tqdm(dl_test, desc=f"{condition}"):
                x = x.to(device)
                z = model.encode(x)

                # 计算 mask 误差
                mean_l1, mean_cos, z_hats, used_masks, act_ratio = compute_masked_error(
                    model, z, mratio, mtype, K, grid_size
                )

                # 计算 base 误差 (与当前评估条件同一 mask_type)
                base_l1, base_cos = compute_base_error(
                    model, z, grid_size, mask_type=mtype
                )

                # 使用对应 mask_type 的 tau
                tau_l1 = tau_dict[mtype]["l1"]
                tau_cos = tau_dict[mtype]["cos"]

                # S_logratio (L1 版本)
                s_lr_l1 = np.log((mean_l1 + tau_l1) / (base_l1 + tau_l1))
                scores_logratio_l1.append(s_lr_l1)

                # S_logratio (cosine 版本)
                s_lr_cos = np.log((mean_cos + tau_cos) / (base_cos + tau_cos))
                scores_logratio_cos.append(s_lr_cos)

                # S_var (使用与预测时完全相同的 mask)
                if len(z_hats) >= 2:
                    s_var = compute_s_var(z_hats, used_masks, grid_size)
                else:
                    s_var = 0.0
                scores_var.append(s_var)
                actual_ratios_all.append(act_ratio)

                labels.append(label.item())

                # 复杂度特征
                cf = compute_complexity_features(x)
                complexity_feats.append(cf)

            labels = np.array(labels)
            scores_l1 = np.array(scores_logratio_l1)
            scores_cos = np.array(scores_logratio_cos)
            s_vars = np.array(scores_var)

            # AUROC / AUPR
            try:
                auroc_l1 = roc_auc_score(labels, scores_l1)
                aupr_l1 = average_precision_score(labels, scores_l1)
            except ValueError:
                auroc_l1 = aupr_l1 = float("nan")

            try:
                auroc_cos = roc_auc_score(labels, scores_cos)
                aupr_cos = average_precision_score(labels, scores_cos)
            except ValueError:
                auroc_cos = aupr_cos = float("nan")

            mean_act_ratio = np.mean(actual_ratios_all)
            log(f"  实际 mask 比例均值: {mean_act_ratio:.4f} (目标 {mratio:.2f})", log_file)
            log(f"  S_logratio(L1):  AUROC={auroc_l1:.4f}, AUPR={aupr_l1:.4f}", log_file)
            log(f"  S_logratio(cos): AUROC={auroc_cos:.4f}, AUPR={aupr_cos:.4f}", log_file)
            log(f"  S_var 均值: real={s_vars[labels == 0].mean():.6f}, "
                f"fake={s_vars[labels == 1].mean():.6f}", log_file)

            # 复杂度相关性分析
            cf_keys = ["edge_density", "hf_energy", "patch_entropy", "grad_energy"]
            for key in cf_keys:
                vals = np.array([cf[key] for cf in complexity_feats])
                if np.std(vals) > 1e-10 and np.std(scores_l1) > 1e-10:
                    corr, pval = pearsonr(vals, scores_l1)
                    log(f"  {key} vs S_logratio(L1): r={corr:.4f}, p={pval:.4f}", log_file)

            all_results[condition] = {
                "auroc_l1": float(auroc_l1),
                "aupr_l1": float(aupr_l1),
                "auroc_cos": float(auroc_cos),
                "aupr_cos": float(aupr_cos),
                "s_var_real_mean": float(s_vars[labels == 0].mean()),
                "s_var_fake_mean": float(s_vars[labels == 1].mean()),
                "tau_l1": float(tau_dict[mtype]["l1"]),
                "tau_cos": float(tau_dict[mtype]["cos"]),
                "actual_mask_ratio_mean": float(mean_act_ratio),
                "K": K,
            }

    # 保存结果
    result_path = os.path.join(eval_dir, f"eval_results_K{K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE Zero-Shot 评估")
    parser.add_argument("--exp_name", type=str, default="A0",
                        help="实验名称")
    parser.add_argument("--K", type=int, default=config.EVAL_K,
                        help="每张图的探测次数")
    args = parser.parse_args()
    evaluate(args)
