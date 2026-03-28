import os
os.environ["OMP_NUM_THREADS"] = "4"
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
from dataset import ChameleonTestDataset, RealPatchDataset  # 5090版本支持 preload
from model import LatentMAE, sample_mask


def setup_cuda():
    """设置 CUDA 优化"""
    if torch.cuda.is_available():
        if getattr(config, "CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def log(msg, log_file=None):
    print(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def load_model(exp_dir, device):
    """加载完整模型"""
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
def batch_encode(model, dataloader, device, desc="编码"):
    """批量编码所有图像, 返回 latents 和 labels"""
    latents = []
    labels = []
    images = []
    
    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)):
            x, label = batch
            labels.extend(label.tolist())
        else:
            x = batch
        
        x = x.to(device, non_blocking=True)
        z = model.encode(x)
        latents.append(z)
        images.append(x)
    
    return latents, labels, images


@torch.no_grad()
def compute_channel_errors_batch(z, z_hat, mask, struct_ch):
    """
    批量计算分通道误差 (全 GPU)
    z, z_hat: [1, C, H, W]
    mask: [N] bool
    返回: struct_l1, struct_cos, detail_l1, detail_cos
    """
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    z_flat = z.flatten(2).permute(0, 2, 1)[0]      # [N, C]
    z_hat_flat = z_hat.flatten(2).permute(0, 2, 1)[0]
    
    z_masked = z_flat[mask]           # [M, C]
    z_hat_masked = z_hat_flat[mask]
    
    # 结构通道
    z_s = z_masked[:, :struct_ch]
    z_hat_s = z_hat_masked[:, :struct_ch]
    struct_l1 = (z_s - z_hat_s).abs().mean().item()
    struct_cos = 1.0 - F.cosine_similarity(z_s, z_hat_s, dim=-1).mean().item()
    
    # 细节通道
    z_d = z_masked[:, struct_ch:]
    z_hat_d = z_hat_masked[:, struct_ch:]
    detail_l1 = (z_d - z_hat_d).abs().mean().item()
    detail_cos = 1.0 - F.cosine_similarity(z_d, z_hat_d, dim=-1).mean().item()
    
    return struct_l1, struct_cos, detail_l1, detail_cos


@torch.no_grad()
def compute_masked_error_batch(model, z, mask_ratio, mask_type, K, grid_size, struct_ch):
    """
    批量 K 次 mask 预测 (一次前向 K 个)
    z: [1, C, H, W]
    返回字典包含分通道误差和 z_hats
    """
    device = z.device
    C, H, W = z.shape[1:]
    
    # 生成 K 个 mask
    masks = []
    actual_ratios = []
    for _ in range(K):
        m, ar = sample_mask(grid_size, mask_ratio, mask_type)
        masks.append(m)
        actual_ratios.append(ar)
    
    masks_tensor = torch.stack(masks).to(device)  # [K, N]
    
    # 批量前向: z 重复 K 次
    z_repeat = z.expand(K, -1, -1, -1)  # [K, C, H, W]
    z_hats = model.predict(z_repeat, masks_tensor)  # [K, C, H, W]
    
    # 分通道计算误差
    struct_l1_list, struct_cos_list = [], []
    detail_l1_list, detail_cos_list = [], []
    
    for k in range(K):
        m = masks_tensor[k]
        if m.sum() == 0:
            continue
        s_l1, s_cos, d_l1, d_cos = compute_channel_errors_batch(
            z, z_hats[k:k+1], m, struct_ch
        )
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
        'z_hats': [z_hats[k:k+1] for k in range(K)],
        'masks': [masks_tensor[k] for k in range(K)],
        'actual_ratio': np.mean(actual_ratios),
    }


@torch.no_grad()
def compute_base_error_batch(model, z, grid_size, mask_type, struct_ch,
                             base_ratio=None, base_runs=None):
    """批量计算 base 误差"""
    if base_ratio is None:
        base_ratio = config.EVAL_BASE_MASK_RATIO
    if base_runs is None:
        base_runs = config.EVAL_BASE_RUNS
    
    device = z.device
    
    masks = []
    for _ in range(base_runs):
        m, _ = sample_mask(grid_size, base_ratio, mask_type)
        masks.append(m)
    
    masks_tensor = torch.stack(masks).to(device)
    z_repeat = z.expand(base_runs, -1, -1, -1)
    z_hats = model.predict(z_repeat, masks_tensor)
    
    struct_l1_list, struct_cos_list = [], []
    detail_l1_list, detail_cos_list = [], []
    
    for k in range(base_runs):
        m = masks_tensor[k]
        if m.sum() == 0:
            continue
        s_l1, s_cos, d_l1, d_cos = compute_channel_errors_batch(z, z_hats[k:k+1], m, struct_ch)
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
    """计算分通道方差 + 交叉方差 (全 GPU)"""
    if len(z_hats) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    N = grid_size * grid_size
    device = z_hats[0].device
    
    z_stack = torch.cat([zh.flatten(2).permute(0, 2, 1) for zh in z_hats], dim=0)  # [K, N, C]
    mask_stack = torch.stack(masks)  # [K, N]
    
    struct_vars = []
    detail_vars = []
    total_vars = []
    cross_covs = []  # 结构-细节协方差
    
    for i in range(N):
        valid_k = mask_stack[:, i]
        if valid_k.sum() < 2:
            continue
        
        preds = z_stack[valid_k, i, :]  # [K', C]
        preds_s = preds[:, :struct_ch]  # [K', 32]
        preds_d = preds[:, struct_ch:]  # [K', 96]
        
        struct_vars.append(preds_s.var(dim=0).mean().item())
        detail_vars.append(preds_d.var(dim=0).mean().item())
        total_vars.append(preds.var(dim=0).mean().item())
        
        # 交叉协方差: 结构均值 vs 细节均值 的协方差
        s_mean = preds_s.mean(dim=1)  # [K']
        d_mean = preds_d.mean(dim=1)  # [K']
        if s_mean.shape[0] >= 2:
            cov = ((s_mean - s_mean.mean()) * (d_mean - d_mean.mean())).mean().abs().item()
            cross_covs.append(cov)
    
    return (
        np.mean(struct_vars) if struct_vars else 0.0,
        np.mean(detail_vars) if detail_vars else 0.0,
        np.mean(total_vars) if total_vars else 0.0,
        np.mean(cross_covs) if cross_covs else 0.0,
    )


@torch.no_grad()
def compute_complexity_features_batch(x):
    """计算复杂度特征 (GPU): edge_density, hf_energy, grad_energy, patch_entropy"""
    gray = x[0].mean(dim=0)
    device = x.device
    
    # Sobel 边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(g, sobel_x, padding=1)
    gy = F.conv2d(g, sobel_y, padding=1)
    edge_density = (gx ** 2 + gy ** 2).sqrt().mean().item()
    
    # 高频能量
    fft = torch.fft.rfft2(gray, norm="ortho")
    mag = fft.abs()
    H, W2 = mag.shape
    fy = torch.arange(H, device=device).float()
    fx = torch.arange(W2, device=device).float()
    fy[fy > H // 2] -= H
    gy_freq, gx_freq = torch.meshgrid(fy, fx, indexing="ij")
    radius = (gy_freq ** 2 + gx_freq ** 2).sqrt()
    hf_mask = radius > 0.5 * radius.max()
    hf_energy = mag[hf_mask].pow(2).sum().item() / (mag.pow(2).sum().item() + 1e-10)
    
    grad_energy = (gx ** 2 + gy ** 2).mean().item()
    
    # Patch entropy: 将图像分成 16x16 块, 计算每块熵的均值
    patch_size = 16
    H_img, W_img = gray.shape
    entropies = []
    for i in range(0, H_img - patch_size + 1, patch_size):
        for j in range(0, W_img - patch_size + 1, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            # 量化到 256 bins
            hist = torch.histc(patch, bins=256, min=0.0, max=1.0)
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            entropy = -(hist * torch.log2(hist)).sum().item()
            entropies.append(entropy)
    patch_entropy = np.mean(entropies) if entropies else 0.0
    
    return {
        "edge_density": edge_density,
        "hf_energy": hf_energy,
        "grad_energy": grad_energy,
        "patch_entropy": patch_entropy,
    }


@torch.no_grad()
def evaluate(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, "eval_log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_cuda()
    log(f"设备: {device}", log_file)

    model = load_model(exp_dir, device)
    grid_size = config.LATENT_SPATIAL
    struct_ch = config.STRUCT_CHANNELS
    K = args.K

    # 批量加载测试集
    batch_size = getattr(config, "EVAL_BATCH_SIZE", 32)
    num_workers = getattr(config, "EVAL_NUM_WORKERS", 4)
    
    # RAM 预加载
    log("预加载图像到 RAM...", log_file)
    ds_test = ChameleonTestDataset(preload=True, num_threads=16)
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    log(f"Chameleon 测试集: {len(ds_test)} 张, batch_size={batch_size}", log_file)

    # 批量编码测试集
    log("批量编码测试集...", log_file)
    test_latents, test_labels, test_images = batch_encode(model, dl_test, device, "编码测试集")
    log(f"测试集编码完成: {len(test_latents)} batches", log_file)

    # 批量编码验证集
    ds_val = RealPatchDataset(patches_per_image=1, split="val", preload=True, num_threads=16)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_latents, _, _ = batch_encode(model, dl_val, device, "编码验证集")
    log(f"验证集编码完成: {len(val_latents)} batches", log_file)

    # 展平 latents
    test_latents_flat = torch.cat(test_latents, dim=0)  # [N_test, C, H, W]
    val_latents_flat = torch.cat(val_latents, dim=0)    # [N_val, C, H, W]
    test_labels_arr = np.array(test_labels)
    log(f"test: {test_latents_flat.shape}, val: {val_latents_flat.shape}", log_file)

    # 计算 tau (分通道)
    mask_types = ["random", "block"]
    eval_ratios = config.EVAL_MASK_RATIOS

    log("计算 tau (L1 + Cos + Var, 按 mtype+ratio 分别统计)...", log_file)
    tau_dict = {}  # tau_dict[mtype][ratio] = {...}
    
    for mtype in mask_types:
        tau_dict[mtype] = {}
        
        # L1/Cos tau: 从 base_error 统计 (与 ratio 无关, 只算一次)
        struct_l1_list, struct_cos_list = [], []
        detail_l1_list, detail_cos_list = [], []
        
        for i in tqdm(range(val_latents_flat.shape[0]), desc=f"tau_base[{mtype}]", leave=False):
            z = val_latents_flat[i:i+1]
            base_err = compute_base_error_batch(model, z, grid_size, mtype, struct_ch)
            struct_l1_list.append(base_err['struct']['l1'])
            struct_cos_list.append(base_err['struct']['cos'])
            detail_l1_list.append(base_err['detail']['l1'])
            detail_cos_list.append(base_err['detail']['cos'])
        
        tau_base = {
            'struct_l1': np.percentile(struct_l1_list, config.TAU_PERCENTILE),
            'struct_cos': np.percentile(struct_cos_list, config.TAU_PERCENTILE),
            'detail_l1': np.percentile(detail_l1_list, config.TAU_PERCENTILE),
            'detail_cos': np.percentile(detail_cos_list, config.TAU_PERCENTILE),
        }
        log(f"  tau_base[{mtype}] L1: struct={tau_base['struct_l1']:.6f}, detail={tau_base['detail_l1']:.6f}", log_file)
        log(f"  tau_base[{mtype}] Cos: struct={tau_base['struct_cos']:.6f}, detail={tau_base['detail_cos']:.6f}", log_file)
        
        # Var tau: 按每个 ratio 分别统计
        for mratio in eval_ratios:
            var_struct_list, var_detail_list, var_cross_list = [], [], []
            
            for i in tqdm(range(val_latents_flat.shape[0]), desc=f"tau_var[{mtype}][{int(mratio*100)}%]", leave=False):
                z = val_latents_flat[i:i+1]
                mask_err = compute_masked_error_batch(
                    model, z, mratio, mtype, K, grid_size, struct_ch
                )
                s_var_s, s_var_d, _, s_var_cross = compute_s_var_channelwise(
                    mask_err['z_hats'], mask_err['masks'], grid_size, struct_ch
                )
                var_struct_list.append(s_var_s)
                var_detail_list.append(s_var_d)
                var_cross_list.append(s_var_cross)
            
            tau_dict[mtype][mratio] = {
                **tau_base,  # 继承 L1/Cos tau
                'var_struct': np.percentile(var_struct_list, config.TAU_PERCENTILE),
                'var_detail': np.percentile(var_detail_list, config.TAU_PERCENTILE),
                'var_cross': np.percentile(var_cross_list, config.TAU_PERCENTILE),
            }
            log(f"  tau_var[{mtype}][{int(mratio*100)}%]: struct={tau_dict[mtype][mratio]['var_struct']:.6f}, "
                f"detail={tau_dict[mtype][mratio]['var_detail']:.6f}", log_file)

    # 预计算 base_error (跨 ratio 复用)
    log("预计算测试集 base_error...", log_file)
    base_errors = {mtype: [] for mtype in mask_types}
    for mtype in mask_types:
        for i in tqdm(range(test_latents_flat.shape[0]), desc=f"base[{mtype}]"):
            z = test_latents_flat[i:i+1]
            base_err = compute_base_error_batch(model, z, grid_size, mtype, struct_ch)
            base_errors[mtype].append(base_err)

    # 预计算复杂度特征
    log("计算复杂度特征...", log_file)
    complexity_feats = []
    for batch_imgs in tqdm(test_images, desc="复杂度"):
        for j in range(batch_imgs.shape[0]):
            complexity_feats.append(compute_complexity_features_batch(batch_imgs[j:j+1]))

    # 评估
    all_results = {}

    for mtype in mask_types:
        for mratio in eval_ratios:
            condition = f"{mtype}_{int(mratio * 100)}%"
            log(f"\n--- 评估条件: {condition}, K={K} ---", log_file)

            # L1 分数
            scores_struct_l1, scores_detail_l1 = [], []
            # Cos 分数
            scores_struct_cos, scores_detail_cos = [], []
            # Var 分数
            scores_var_struct, scores_var_detail, scores_var_total, scores_var_cross = [], [], [], []
            # Couple
            scores_couple = []
            actual_ratios_all = []

            tau = tau_dict[mtype][mratio]  # 按 mtype + ratio 获取 tau

            for i in tqdm(range(test_latents_flat.shape[0]), desc=f"{condition}"):
                z = test_latents_flat[i:i+1]
                
                mask_err = compute_masked_error_batch(
                    model, z, mratio, mtype, K, grid_size, struct_ch
                )
                base_err = base_errors[mtype][i]

                # S_struct_l1
                s_struct_l1 = np.log(
                    (mask_err['struct']['l1'] + tau['struct_l1']) /
                    (base_err['struct']['l1'] + tau['struct_l1'])
                )
                scores_struct_l1.append(s_struct_l1)

                # S_struct_cos
                s_struct_cos = np.log(
                    (mask_err['struct']['cos'] + tau['struct_cos']) /
                    (base_err['struct']['cos'] + tau['struct_cos'])
                )
                scores_struct_cos.append(s_struct_cos)

                # S_detail_l1
                s_detail_l1 = np.log(
                    (mask_err['detail']['l1'] + tau['detail_l1']) /
                    (base_err['detail']['l1'] + tau['detail_l1'])
                )
                scores_detail_l1.append(s_detail_l1)

                # S_detail_cos
                s_detail_cos = np.log(
                    (mask_err['detail']['cos'] + tau['detail_cos']) /
                    (base_err['detail']['cos'] + tau['detail_cos'])
                )
                scores_detail_cos.append(s_detail_cos)

                # S_var (分通道 + cross)
                s_var_s, s_var_d, s_var_t, s_var_c = compute_s_var_channelwise(
                    mask_err['z_hats'], mask_err['masks'], grid_size, struct_ch
                )
                scores_var_struct.append(s_var_s)
                scores_var_detail.append(s_var_d)
                scores_var_total.append(s_var_t)
                scores_var_cross.append(s_var_c)

                # S_couple-lite: 使用 var tau (量纲一致, 按 ratio 校准)
                s_couple = np.log(
                    (s_var_d + tau['var_detail']) / (s_var_s + tau['var_struct'] + 1e-10)
                )
                scores_couple.append(s_couple)
                actual_ratios_all.append(mask_err['actual_ratio'])

            # 转换为数组
            scores_struct_l1 = np.array(scores_struct_l1)
            scores_struct_cos = np.array(scores_struct_cos)
            scores_detail_l1 = np.array(scores_detail_l1)
            scores_detail_cos = np.array(scores_detail_cos)
            scores_couple = np.array(scores_couple)
            s_var_struct_arr = np.array(scores_var_struct)
            s_var_detail_arr = np.array(scores_var_detail)
            s_var_total_arr = np.array(scores_var_total)
            s_var_cross_arr = np.array(scores_var_cross)
            labels = test_labels_arr

            # AUROC / AUPR
            def safe_auroc_aupr(y_true, y_score):
                try:
                    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
                except:
                    return float("nan"), float("nan")

            # L1 分数 AUROC/AUPR
            auroc_struct_l1, aupr_struct_l1 = safe_auroc_aupr(labels, scores_struct_l1)
            auroc_detail_l1, aupr_detail_l1 = safe_auroc_aupr(labels, scores_detail_l1)
            # Cos 分数 AUROC/AUPR
            auroc_struct_cos, aupr_struct_cos = safe_auroc_aupr(labels, scores_struct_cos)
            auroc_detail_cos, aupr_detail_cos = safe_auroc_aupr(labels, scores_detail_cos)
            # Couple + Var
            auroc_couple, aupr_couple = safe_auroc_aupr(labels, scores_couple)
            auroc_var_s, aupr_var_s = safe_auroc_aupr(labels, s_var_struct_arr)
            auroc_var_d, aupr_var_d = safe_auroc_aupr(labels, s_var_detail_arr)
            auroc_var_t, aupr_var_t = safe_auroc_aupr(labels, s_var_total_arr)
            auroc_var_c, aupr_var_c = safe_auroc_aupr(labels, s_var_cross_arr)

            mean_act_ratio = np.mean(actual_ratios_all)
            log(f"  实际 mask 比例: {mean_act_ratio:.4f}", log_file)
            log(f"  S_struct_l1:   AUROC={auroc_struct_l1:.4f}, AUPR={aupr_struct_l1:.4f}", log_file)
            log(f"  S_struct_cos:  AUROC={auroc_struct_cos:.4f}, AUPR={aupr_struct_cos:.4f}", log_file)
            log(f"  S_detail_l1:   AUROC={auroc_detail_l1:.4f}, AUPR={aupr_detail_l1:.4f}", log_file)
            log(f"  S_detail_cos:  AUROC={auroc_detail_cos:.4f}, AUPR={aupr_detail_cos:.4f}", log_file)
            log(f"  S_couple-lite: AUROC={auroc_couple:.4f}, AUPR={aupr_couple:.4f}", log_file)
            log(f"  S_var_struct:  AUROC={auroc_var_s:.4f}, AUPR={aupr_var_s:.4f}", log_file)
            log(f"  S_var_detail:  AUROC={auroc_var_d:.4f}, AUPR={aupr_var_d:.4f}", log_file)
            log(f"  S_var_cross:   AUROC={auroc_var_c:.4f}, AUPR={aupr_var_c:.4f}", log_file)
            log(f"  S_var (total): AUROC={auroc_var_t:.4f}, AUPR={aupr_var_t:.4f}", log_file)

            # 均值统计
            log(f"  S_struct_l1 均值: real={scores_struct_l1[labels==0].mean():.6f}, fake={scores_struct_l1[labels==1].mean():.6f}", log_file)
            log(f"  S_detail_l1 均值: real={scores_detail_l1[labels==0].mean():.6f}, fake={scores_detail_l1[labels==1].mean():.6f}", log_file)

            # 复杂度相关性 (扩展到多个分数)
            complexity_keys = ["edge_density", "hf_energy", "grad_energy", "patch_entropy"]
            score_dict = {
                "S_struct_l1": scores_struct_l1,
                "S_struct_cos": scores_struct_cos,
                "S_detail_l1": scores_detail_l1,
                "S_detail_cos": scores_detail_cos,
                "S_couple": scores_couple,
                "S_var_detail": s_var_detail_arr,
            }
            for ckey in complexity_keys:
                vals = np.array([cf[ckey] for cf in complexity_feats])
                if np.std(vals) < 1e-10:
                    continue
                for sname, sarr in score_dict.items():
                    if np.std(sarr) > 1e-10:
                        corr, pval = pearsonr(vals, sarr)
                        log(f"  {ckey} vs {sname}: r={corr:.4f}, p={pval:.4f}", log_file)

            all_results[condition] = {
                "auroc_struct_l1": float(auroc_struct_l1),
                "aupr_struct_l1": float(aupr_struct_l1),
                "auroc_struct_cos": float(auroc_struct_cos),
                "aupr_struct_cos": float(aupr_struct_cos),
                "auroc_detail_l1": float(auroc_detail_l1),
                "aupr_detail_l1": float(aupr_detail_l1),
                "auroc_detail_cos": float(auroc_detail_cos),
                "aupr_detail_cos": float(aupr_detail_cos),
                "auroc_couple": float(auroc_couple),
                "aupr_couple": float(aupr_couple),
                "auroc_var_struct": float(auroc_var_s),
                "aupr_var_struct": float(aupr_var_s),
                "auroc_var_detail": float(auroc_var_d),
                "aupr_var_detail": float(aupr_var_d),
                "auroc_var_cross": float(auroc_var_c),
                "aupr_var_cross": float(aupr_var_c),
                "auroc_var_total": float(auroc_var_t),
                "aupr_var_total": float(aupr_var_t),
                "s_struct_l1_real_mean": float(scores_struct_l1[labels==0].mean()),
                "s_struct_l1_fake_mean": float(scores_struct_l1[labels==1].mean()),
                "s_detail_l1_real_mean": float(scores_detail_l1[labels==0].mean()),
                "s_detail_l1_fake_mean": float(scores_detail_l1[labels==1].mean()),
                "tau_struct_l1": float(tau['struct_l1']),
                "tau_struct_cos": float(tau['struct_cos']),
                "tau_detail_l1": float(tau['detail_l1']),
                "tau_detail_cos": float(tau['detail_cos']),
                "tau_var_struct": float(tau['var_struct']),
                "tau_var_detail": float(tau['var_detail']),
                "actual_mask_ratio_mean": float(mean_act_ratio),
                "K": K,
            }

    # 保存
    result_path = os.path.join(eval_dir, f"eval_results_K{K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE 评估 (5090 优化版)")
    parser.add_argument("--exp_name", type=str, default="A0", help="实验名称")
    parser.add_argument("--K", type=int, default=config.EVAL_K, help="探测次数")
    args = parser.parse_args()
    evaluate(args)
