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
from dataset import ChameleonTestDataset, RealPatchDataset
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
    
    struct_l1_list, detail_l1_list = [], []
    
    for k in range(base_runs):
        m = masks_tensor[k]
        if m.sum() == 0:
            continue
        s_l1, _, d_l1, _ = compute_channel_errors_batch(z, z_hats[k:k+1], m, struct_ch)
        struct_l1_list.append(s_l1)
        detail_l1_list.append(d_l1)
    
    return {
        'struct': {'l1': np.mean(struct_l1_list) if struct_l1_list else 0.0},
        'detail': {'l1': np.mean(detail_l1_list) if detail_l1_list else 0.0},
    }


@torch.no_grad()
def compute_s_var_channelwise(z_hats, masks, grid_size, struct_ch):
    """计算分通道方差 (全 GPU)"""
    if len(z_hats) < 2:
        return 0.0, 0.0, 0.0
    
    N = grid_size * grid_size
    device = z_hats[0].device
    
    # 堆叠所有 z_hat: [K, N, C]
    z_stack = torch.cat([zh.flatten(2).permute(0, 2, 1) for zh in z_hats], dim=0)  # [K, N, C]
    mask_stack = torch.stack(masks)  # [K, N]
    
    struct_vars = []
    detail_vars = []
    total_vars = []
    
    for i in range(N):
        valid_k = mask_stack[:, i]  # [K] bool
        if valid_k.sum() < 2:
            continue
        
        preds = z_stack[valid_k, i, :]  # [K', C]
        struct_vars.append(preds[:, :struct_ch].var(dim=0).mean().item())
        detail_vars.append(preds[:, struct_ch:].var(dim=0).mean().item())
        total_vars.append(preds.var(dim=0).mean().item())
    
    return (
        np.mean(struct_vars) if struct_vars else 0.0,
        np.mean(detail_vars) if detail_vars else 0.0,
        np.mean(total_vars) if total_vars else 0.0,
    )


@torch.no_grad()
def compute_complexity_features_batch(x):
    """计算复杂度特征 (GPU)"""
    gray = x[0].mean(dim=0)
    device = x.device
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(g, sobel_x, padding=1)
    gy = F.conv2d(g, sobel_y, padding=1)
    edge_density = (gx ** 2 + gy ** 2).sqrt().mean().item()
    
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
    
    return {
        "edge_density": edge_density,
        "hf_energy": hf_energy,
        "grad_energy": grad_energy,
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
    
    ds_test = ChameleonTestDataset()
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
    ds_val = RealPatchDataset(patches_per_image=1, split="val")
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

    log("计算 tau (分通道, 批量)...", log_file)
    tau_dict = {}
    for mtype in mask_types:
        struct_l1_list, detail_l1_list = [], []
        for i in tqdm(range(val_latents_flat.shape[0]), desc=f"tau[{mtype}]", leave=False):
            z = val_latents_flat[i:i+1]
            base_err = compute_base_error_batch(model, z, grid_size, mtype, struct_ch)
            struct_l1_list.append(base_err['struct']['l1'])
            detail_l1_list.append(base_err['detail']['l1'])
        
        tau_dict[mtype] = {
            'struct_l1': np.percentile(struct_l1_list, config.TAU_PERCENTILE),
            'detail_l1': np.percentile(detail_l1_list, config.TAU_PERCENTILE),
        }
        log(f"  tau[{mtype}]: struct={tau_dict[mtype]['struct_l1']:.6f}, "
            f"detail={tau_dict[mtype]['detail_l1']:.6f}", log_file)

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

            scores_struct, scores_detail, scores_couple = [], [], []
            scores_var_struct, scores_var_detail, scores_var_total = [], [], []
            actual_ratios_all = []

            for i in tqdm(range(test_latents_flat.shape[0]), desc=f"{condition}"):
                z = test_latents_flat[i:i+1]
                
                mask_err = compute_masked_error_batch(
                    model, z, mratio, mtype, K, grid_size, struct_ch
                )
                base_err = base_errors[mtype][i]
                tau = tau_dict[mtype]

                # S_struct
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

                # S_var
                s_var_s, s_var_d, s_var_t = compute_s_var_channelwise(
                    mask_err['z_hats'], mask_err['masks'], grid_size, struct_ch
                )
                scores_var_struct.append(s_var_s)
                scores_var_detail.append(s_var_d)
                scores_var_total.append(s_var_t)

                # S_couple-lite
                s_couple = np.log(
                    (s_var_d + tau['detail_l1']) / (s_var_s + tau['struct_l1'] + 1e-10)
                )
                scores_couple.append(s_couple)
                actual_ratios_all.append(mask_err['actual_ratio'])

            # 转换为数组
            scores_struct = np.array(scores_struct)
            scores_detail = np.array(scores_detail)
            scores_couple = np.array(scores_couple)
            s_var_struct_arr = np.array(scores_var_struct)
            s_var_detail_arr = np.array(scores_var_detail)
            s_var_total_arr = np.array(scores_var_total)
            labels = test_labels_arr

            # AUROC / AUPR
            def safe_auroc_aupr(y_true, y_score):
                try:
                    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
                except:
                    return float("nan"), float("nan")

            auroc_struct, aupr_struct = safe_auroc_aupr(labels, scores_struct)
            auroc_detail, aupr_detail = safe_auroc_aupr(labels, scores_detail)
            auroc_couple, aupr_couple = safe_auroc_aupr(labels, scores_couple)
            auroc_var_s, aupr_var_s = safe_auroc_aupr(labels, s_var_struct_arr)
            auroc_var_d, aupr_var_d = safe_auroc_aupr(labels, s_var_detail_arr)
            auroc_var_t, aupr_var_t = safe_auroc_aupr(labels, s_var_total_arr)

            mean_act_ratio = np.mean(actual_ratios_all)
            log(f"  实际 mask 比例: {mean_act_ratio:.4f}", log_file)
            log(f"  S_struct:      AUROC={auroc_struct:.4f}, AUPR={aupr_struct:.4f}", log_file)
            log(f"  S_detail:      AUROC={auroc_detail:.4f}, AUPR={aupr_detail:.4f}", log_file)
            log(f"  S_couple-lite: AUROC={auroc_couple:.4f}, AUPR={aupr_couple:.4f}", log_file)
            log(f"  S_var_struct:  AUROC={auroc_var_s:.4f}, AUPR={aupr_var_s:.4f}", log_file)
            log(f"  S_var_detail:  AUROC={auroc_var_d:.4f}, AUPR={aupr_var_d:.4f}", log_file)
            log(f"  S_var (total): AUROC={auroc_var_t:.4f}, AUPR={aupr_var_t:.4f}", log_file)

            # 均值统计
            log(f"  S_struct 均值: real={scores_struct[labels==0].mean():.6f}, fake={scores_struct[labels==1].mean():.6f}", log_file)
            log(f"  S_detail 均值: real={scores_detail[labels==0].mean():.6f}, fake={scores_detail[labels==1].mean():.6f}", log_file)

            # 复杂度相关性
            for key in ["edge_density", "hf_energy", "grad_energy"]:
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
                "s_struct_real_mean": float(scores_struct[labels==0].mean()),
                "s_struct_fake_mean": float(scores_struct[labels==1].mean()),
                "s_detail_real_mean": float(scores_detail[labels==0].mean()),
                "s_detail_fake_mean": float(scores_detail[labels==1].mean()),
                "tau_struct_l1": float(tau['struct_l1']),
                "tau_detail_l1": float(tau['detail_l1']),
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
