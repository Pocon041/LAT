import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
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
    """加载完整模型 (AE + 辅助头 + Predictor)"""
    model = LatentMAE(latent_channels=config.LATENT_CHANNELS).to(device)

    ae_path = os.path.join(exp_dir, "best_ae.pth")
    pred_path = os.path.join(exp_dir, "best_predictor.pth")

    ae_ckpt = torch.load(ae_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(ae_ckpt["encoder"])
    model.decoder.load_state_dict(ae_ckpt["decoder"])
    if "struct_head" in ae_ckpt:
        model.struct_head.load_state_dict(ae_ckpt["struct_head"])
    if "detail_head" in ae_ckpt:
        model.detail_head.load_state_dict(ae_ckpt["detail_head"])
    print(f"加载 AE: {ae_path} (epoch={ae_ckpt['epoch']})")

    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=True)
    model.predictor.load_state_dict(pred_ckpt["predictor"])
    print(f"加载 Predictor: {pred_path} (epoch={pred_ckpt['epoch']})")

    model.eval()
    return model


@torch.no_grad()
def compute_channel_error(z, z_hat, mask, channel_slice):
    """
    计算指定通道范围内的 masked L1 和 cosine error
    z, z_hat: [B, C, H, W]
    mask: [N] bool
    channel_slice: slice object
    返回: l1_err, cos_err
    """
    z_ch = z[:, channel_slice]
    z_hat_ch = z_hat[:, channel_slice]
    
    z_flat = z_ch.flatten(2).permute(0, 2, 1)[0]      # [N, C_sub]
    z_hat_flat = z_hat_ch.flatten(2).permute(0, 2, 1)[0]
    
    if mask.sum() == 0:
        return 0.0, 0.0
    
    z_masked = z_flat[mask]
    z_hat_masked = z_hat_flat[mask]
    
    l1_err = (z_masked - z_hat_masked).abs().mean().item()
    cos_sim = F.cosine_similarity(z_masked, z_hat_masked, dim=-1).mean().item()
    cos_err = 1.0 - cos_sim
    
    return l1_err, cos_err


@torch.no_grad()
def compute_masked_error_channelwise(model, z, mask_ratio, mask_type, K, grid_size, struct_ch):
    """
    对单个 latent z 做 K 次不同 mask 的补全, 分通道计算误差
    返回: {
        'struct': {'l1': mean, 'cos': mean},
        'detail': {'l1': mean, 'cos': mean},
        'z_hats': list of z_hat tensors,
        'masks': list of mask tensors,
        'actual_ratio': float
    }
    """
    struct_slice = slice(0, struct_ch)
    detail_slice = slice(struct_ch, None)
    
    struct_l1_list, struct_cos_list = [], []
    detail_l1_list, detail_cos_list = [], []
    z_hats = []
    used_masks = []
    actual_ratios = []
    
    for _ in range(K):
        mask, actual_ratio = sample_mask(grid_size, mask_ratio, mask_type)
        actual_ratios.append(actual_ratio)
        mask = mask.to(z.device)
        mask_batch = mask.unsqueeze(0)
        
        z_hat = model.predict(z, mask_batch)
        
        if mask.sum() == 0:
            continue
        
        # 结构通道误差
        s_l1, s_cos = compute_channel_error(z, z_hat, mask, struct_slice)
        struct_l1_list.append(s_l1)
        struct_cos_list.append(s_cos)
        
        # 细节通道误差
        d_l1, d_cos = compute_channel_error(z, z_hat, mask, detail_slice)
        detail_l1_list.append(d_l1)
        detail_cos_list.append(d_cos)
        
        z_hats.append(z_hat)
        used_masks.append(mask)
    
    return {
        'struct': {
            'l1': np.mean(struct_l1_list) if struct_l1_list else 0.0,
            'cos': np.mean(struct_cos_list) if struct_cos_list else 0.0,
        },
        'detail': {
            'l1': np.mean(detail_l1_list) if detail_l1_list else 0.0,
            'cos': np.mean(detail_cos_list) if detail_cos_list else 0.0,
        },
        'z_hats': z_hats,
        'masks': used_masks,
        'actual_ratio': np.mean(actual_ratios) if actual_ratios else mask_ratio,
    }


@torch.no_grad()
def compute_base_error_channelwise(model, z, grid_size, mask_type, struct_ch,
                                   base_ratio=None, base_runs=None):
    """计算 base 误差 (分通道)"""
    if base_ratio is None:
        base_ratio = config.EVAL_BASE_MASK_RATIO
    if base_runs is None:
        base_runs = config.EVAL_BASE_RUNS
    
    struct_slice = slice(0, struct_ch)
    detail_slice = slice(struct_ch, None)
    
    struct_l1_list, struct_cos_list = [], []
    detail_l1_list, detail_cos_list = [], []
    
    for _ in range(base_runs):
        mask, _ = sample_mask(grid_size, base_ratio, mask_type)
        mask = mask.to(z.device)
        mask_batch = mask.unsqueeze(0)
        
        z_hat = model.predict(z, mask_batch)
        
        if mask.sum() == 0:
            continue
        
        s_l1, s_cos = compute_channel_error(z, z_hat, mask, struct_slice)
        d_l1, d_cos = compute_channel_error(z, z_hat, mask, detail_slice)
        
        struct_l1_list.append(s_l1)
        struct_cos_list.append(s_cos)
        detail_l1_list.append(d_l1)
        detail_cos_list.append(d_cos)
    
    return {
        'struct': {
            'l1': np.mean(struct_l1_list) if struct_l1_list else 0.0,
            'cos': np.mean(struct_cos_list) if struct_cos_list else 0.0,
        },
        'detail': {
            'l1': np.mean(detail_l1_list) if detail_l1_list else 0.0,
            'cos': np.mean(detail_cos_list) if detail_cos_list else 0.0,
        },
    }


@torch.no_grad()
def compute_s_var_channelwise(z_hats, masks, grid_size, struct_ch):
    """
    计算 S_var (分通道): 对每个 token, 只统计该 token 被 mask 时的预测方差
    返回: s_var_struct, s_var_detail, s_var_total
    """
    if len(z_hats) < 2:
        return 0.0, 0.0, 0.0
    
    N = grid_size * grid_size
    struct_slice = slice(0, struct_ch)
    detail_slice = slice(struct_ch, None)
    
    struct_vars = []
    detail_vars = []
    total_vars = []
    
    for i in range(N):
        preds_struct = []
        preds_detail = []
        preds_total = []
        
        for z_hat, mask in zip(z_hats, masks):
            if mask[i]:
                z_flat = z_hat.flatten(2).permute(0, 2, 1)[0]  # [N, C]
                preds_struct.append(z_flat[i, :struct_ch])
                preds_detail.append(z_flat[i, struct_ch:])
                preds_total.append(z_flat[i])
        
        if len(preds_struct) >= 2:
            struct_vars.append(torch.stack(preds_struct).var(dim=0).mean().item())
            detail_vars.append(torch.stack(preds_detail).var(dim=0).mean().item())
            total_vars.append(torch.stack(preds_total).var(dim=0).mean().item())
    
    s_var_struct = np.mean(struct_vars) if struct_vars else 0.0
    s_var_detail = np.mean(detail_vars) if detail_vars else 0.0
    s_var_total = np.mean(total_vars) if total_vars else 0.0
    
    return s_var_struct, s_var_detail, s_var_total


@torch.no_grad()
def compute_complexity_features(img_tensor):
    """计算图像复杂度特征"""
    x = img_tensor[0]
    gray = x.mean(dim=0)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(g, sobel_x, padding=1)
    gy = F.conv2d(g, sobel_y, padding=1)
    edge_density = (gx ** 2 + gy ** 2).sqrt().mean().item()

    fft = torch.fft.rfft2(gray, norm="ortho")
    mag = fft.abs()
    H, W2 = mag.shape
    fy = torch.arange(H, device=x.device).float()
    fx = torch.arange(W2, device=x.device).float()
    fy[fy > H // 2] -= H
    gy_freq, gx_freq = torch.meshgrid(fy, fx, indexing="ij")
    radius = (gy_freq ** 2 + gx_freq ** 2).sqrt()
    max_r = radius.max()
    hf_mask = radius > 0.5 * max_r
    hf_energy = mag[hf_mask].pow(2).sum().item() / (mag.pow(2).sum().item() + 1e-10)

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

    model = load_model(exp_dir, device)
    grid_size = config.LATENT_SPATIAL
    struct_ch = config.STRUCT_CHANNELS
    K = args.K

    ds_test = ChameleonTestDataset()
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2)
    log(f"Chameleon 测试集: {len(ds_test)} 张", log_file)

    # --- 计算 tau (分通道, 按 mask_type 分别统计) ---
    mask_types = ["random", "block"]
    eval_ratios = config.EVAL_MASK_RATIOS

    log("计算 tau (基于验证集 base 误差, 分通道按 mask_type 统计)...", log_file)
    ds_val = RealPatchDataset(patches_per_image=1, split="val")
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=2)

    val_latents = []
    for batch in tqdm(dl_val, desc="提取验证集 latent"):
        x = batch.to(device)
        z = model.encode(x)
        val_latents.append(z)

    tau_dict = {}
    for mtype in mask_types:
        struct_l1_list, struct_cos_list = [], []
        detail_l1_list, detail_cos_list = [], []
        
        for z in tqdm(val_latents, desc=f"  tau[{mtype}]", leave=False):
            base_err = compute_base_error_channelwise(
                model, z, grid_size, mtype, struct_ch
            )
            struct_l1_list.append(base_err['struct']['l1'])
            struct_cos_list.append(base_err['struct']['cos'])
            detail_l1_list.append(base_err['detail']['l1'])
            detail_cos_list.append(base_err['detail']['cos'])
        
        tau_dict[mtype] = {
            'struct_l1': np.percentile(struct_l1_list, config.TAU_PERCENTILE),
            'struct_cos': np.percentile(struct_cos_list, config.TAU_PERCENTILE),
            'detail_l1': np.percentile(detail_l1_list, config.TAU_PERCENTILE),
            'detail_cos': np.percentile(detail_cos_list, config.TAU_PERCENTILE),
        }
        log(f"  tau[{mtype}]: struct_l1={tau_dict[mtype]['struct_l1']:.6f}, "
            f"detail_l1={tau_dict[mtype]['detail_l1']:.6f}", log_file)

    # --- 评估 ---
    all_results = {}

    for mtype in mask_types:
        for mratio in eval_ratios:
            condition = f"{mtype}_{int(mratio * 100)}%"
            log(f"\n--- 评估条件: {condition}, K={K} ---", log_file)

            # 收集分数
            scores_struct = []      # S_struct
            scores_detail = []      # S_detail
            scores_couple = []      # S_couple-lite
            scores_var_struct = []  # S_var_struct
            scores_var_detail = []  # S_var_detail
            scores_var_total = []   # S_var (全通道)
            labels = []
            complexity_feats = []
            actual_ratios_all = []

            for x, label in tqdm(dl_test, desc=f"{condition}"):
                x = x.to(device)
                z = model.encode(x)

                # 计算 mask 误差 (分通道)
                mask_err = compute_masked_error_channelwise(
                    model, z, mratio, mtype, K, grid_size, struct_ch
                )

                # 计算 base 误差 (分通道)
                base_err = compute_base_error_channelwise(
                    model, z, grid_size, mtype, struct_ch
                )

                tau = tau_dict[mtype]

                # S_struct = log((Err_mask_struct + tau_s) / (Err_base_struct + tau_s))
                s_struct = np.log(
                    (mask_err['struct']['l1'] + tau['struct_l1']) /
                    (base_err['struct']['l1'] + tau['struct_l1'])
                )
                scores_struct.append(s_struct)

                # S_detail
                s_detail = np.log(
                    (mask_err['detail']['l1'] + tau['detail_l1']) /
                    (base_err['detail']['l1'] + tau['detail_l1'])
                )
                scores_detail.append(s_detail)

                # S_var (分通道)
                s_var_s, s_var_d, s_var_t = compute_s_var_channelwise(
                    mask_err['z_hats'], mask_err['masks'], grid_size, struct_ch
                )
                scores_var_struct.append(s_var_s)
                scores_var_detail.append(s_var_d)
                scores_var_total.append(s_var_t)

                # S_couple-lite = log((Var_detail + tau_d) / (Var_struct + tau_s))
                s_couple = np.log(
                    (s_var_d + tau['detail_l1']) / (s_var_s + tau['struct_l1'] + 1e-10)
                )
                scores_couple.append(s_couple)

                actual_ratios_all.append(mask_err['actual_ratio'])
                labels.append(label.item())
                complexity_feats.append(compute_complexity_features(x))

            labels = np.array(labels)
            scores_struct = np.array(scores_struct)
            scores_detail = np.array(scores_detail)
            scores_couple = np.array(scores_couple)
            s_var_struct_arr = np.array(scores_var_struct)
            s_var_detail_arr = np.array(scores_var_detail)
            s_var_total_arr = np.array(scores_var_total)

            # 计算 AUROC / AUPR
            def safe_auroc_aupr(y_true, y_score):
                try:
                    auroc = roc_auc_score(y_true, y_score)
                    aupr = average_precision_score(y_true, y_score)
                except ValueError:
                    auroc = aupr = float("nan")
                return auroc, aupr

            auroc_struct, aupr_struct = safe_auroc_aupr(labels, scores_struct)
            auroc_detail, aupr_detail = safe_auroc_aupr(labels, scores_detail)
            auroc_couple, aupr_couple = safe_auroc_aupr(labels, scores_couple)
            auroc_var_s, aupr_var_s = safe_auroc_aupr(labels, s_var_struct_arr)
            auroc_var_d, aupr_var_d = safe_auroc_aupr(labels, s_var_detail_arr)
            auroc_var_t, aupr_var_t = safe_auroc_aupr(labels, s_var_total_arr)

            mean_act_ratio = np.mean(actual_ratios_all)
            log(f"  实际 mask 比例均值: {mean_act_ratio:.4f} (目标 {mratio:.2f})", log_file)
            log(f"  S_struct:       AUROC={auroc_struct:.4f}, AUPR={aupr_struct:.4f}", log_file)
            log(f"  S_detail:       AUROC={auroc_detail:.4f}, AUPR={aupr_detail:.4f}", log_file)
            log(f"  S_couple-lite:  AUROC={auroc_couple:.4f}, AUPR={aupr_couple:.4f}", log_file)
            log(f"  S_var_struct:   AUROC={auroc_var_s:.4f}, AUPR={aupr_var_s:.4f}", log_file)
            log(f"  S_var_detail:   AUROC={auroc_var_d:.4f}, AUPR={aupr_var_d:.4f}", log_file)
            log(f"  S_var (total):  AUROC={auroc_var_t:.4f}, AUPR={aupr_var_t:.4f}", log_file)

            # 均值统计
            log(f"  S_struct 均值: real={scores_struct[labels == 0].mean():.6f}, "
                f"fake={scores_struct[labels == 1].mean():.6f}", log_file)
            log(f"  S_detail 均值: real={scores_detail[labels == 0].mean():.6f}, "
                f"fake={scores_detail[labels == 1].mean():.6f}", log_file)
            log(f"  S_var_struct 均值: real={s_var_struct_arr[labels == 0].mean():.6f}, "
                f"fake={s_var_struct_arr[labels == 1].mean():.6f}", log_file)
            log(f"  S_var_detail 均值: real={s_var_detail_arr[labels == 0].mean():.6f}, "
                f"fake={s_var_detail_arr[labels == 1].mean():.6f}", log_file)

            # 复杂度相关性分析
            cf_keys = ["edge_density", "hf_energy", "patch_entropy", "grad_energy"]
            for key in cf_keys:
                vals = np.array([cf[key] for cf in complexity_feats])
                if np.std(vals) > 1e-10 and np.std(scores_struct) > 1e-10:
                    corr, pval = pearsonr(vals, scores_struct)
                    log(f"  {key} vs S_struct: r={corr:.4f}, p={pval:.4f}", log_file)

            all_results[condition] = {
                "auroc_struct": float(auroc_struct),
                "aupr_struct": float(aupr_struct),
                "auroc_detail": float(auroc_detail),
                "aupr_detail": float(aupr_detail),
                "auroc_couple": float(auroc_couple),
                "aupr_couple": float(aupr_couple),
                "auroc_var_struct": float(auroc_var_s),
                "aupr_var_struct": float(aupr_var_s),
                "auroc_var_detail": float(auroc_var_d),
                "aupr_var_detail": float(aupr_var_d),
                "auroc_var_total": float(auroc_var_t),
                "aupr_var_total": float(aupr_var_t),
                "s_struct_real_mean": float(scores_struct[labels == 0].mean()),
                "s_struct_fake_mean": float(scores_struct[labels == 1].mean()),
                "s_detail_real_mean": float(scores_detail[labels == 0].mean()),
                "s_detail_fake_mean": float(scores_detail[labels == 1].mean()),
                "s_var_struct_real_mean": float(s_var_struct_arr[labels == 0].mean()),
                "s_var_struct_fake_mean": float(s_var_struct_arr[labels == 1].mean()),
                "s_var_detail_real_mean": float(s_var_detail_arr[labels == 0].mean()),
                "s_var_detail_fake_mean": float(s_var_detail_arr[labels == 1].mean()),
                "tau_struct_l1": float(tau['struct_l1']),
                "tau_detail_l1": float(tau['detail_l1']),
                "actual_mask_ratio_mean": float(mean_act_ratio),
                "K": K,
            }

    # 保存结果
    result_path = os.path.join(eval_dir, f"eval_results_K{K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE Zero-Shot 评估 (Structure-Aware)")
    parser.add_argument("--exp_name", type=str, default="A0", help="实验名称")
    parser.add_argument("--K", type=int, default=config.EVAL_K, help="每张图的探测次数")
    args = parser.parse_args()
    evaluate(args)
