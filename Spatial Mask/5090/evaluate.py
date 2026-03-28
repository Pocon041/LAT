import os
os.environ["OMP_NUM_THREADS"] = "4"
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from tqdm import tqdm

import config as config
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
def batch_encode(model, dataloader, device, desc="编码"):
    """批量编码, 返回 (latents_list, labels_list, images_list)"""
    latents = []
    labels = []
    images = []
    for batch in tqdm(dataloader, desc=desc):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, lbl = batch
            x = x.to(device, non_blocking=True)
            z = model.encode(x)
            for i in range(x.size(0)):
                latents.append(z[i:i+1])
                labels.append(lbl[i].item())
                images.append(x[i:i+1])
        else:
            x = batch.to(device, non_blocking=True)
            z = model.encode(x)
            for i in range(x.size(0)):
                latents.append(z[i:i+1])
                images.append(x[i:i+1])
    return latents, labels, images


@torch.no_grad()
def compute_masked_error(model, z, mask_ratio, mask_type, K, grid_size):
    """
    对单个 latent z 做 K 次不同 mask 的补全, batch forward 一次完成
    返回: mean_l1_err, mean_cos_err, per_run_z_hats, per_run_masks, mean_actual_ratio
    """
    B, C, H, W = z.shape
    N = H * W

    # 生成 K 个 mask 并堆叠
    masks = []
    actual_ratios = []
    for _ in range(K):
        m, ar = sample_mask(grid_size, mask_ratio, mask_type)
        masks.append(m)
        actual_ratios.append(ar)
    masks_batch = torch.stack(masks).to(z.device)  # [K, N]

    # z 重复 K 份, 一次 batch forward
    z_repeat = z.expand(K, -1, -1, -1)  # [K, C, H, W]
    z_hats = model.predict(z_repeat, masks_batch)  # [K, C, H, W]

    # 计算误差
    z_flat = z.flatten(2).permute(0, 2, 1).expand(K, -1, -1)  # [K, N, C]
    z_hats_flat = z_hats.flatten(2).permute(0, 2, 1)  # [K, N, C]

    l1_errors = []
    cos_errors = []
    used_z_hats = []
    used_masks = []

    for k in range(K):
        m = masks_batch[k]  # [N]
        if m.sum() == 0:
            continue
        z_m = z_flat[k, m]
        z_hat_m = z_hats_flat[k, m]

        l1_errors.append((z_m - z_hat_m).abs().mean().item())
        cos_sim = torch.nn.functional.cosine_similarity(
            z_m, z_hat_m, dim=-1
        ).mean().item()
        cos_errors.append(1.0 - cos_sim)

        used_z_hats.append(z_hats_flat[k:k+1])
        used_masks.append(m)

    mean_l1 = np.mean(l1_errors) if l1_errors else 0.0
    mean_cos = np.mean(cos_errors) if cos_errors else 0.0
    mean_actual_ratio = np.mean(actual_ratios) if actual_ratios else mask_ratio
    return mean_l1, mean_cos, used_z_hats, used_masks, mean_actual_ratio


@torch.no_grad()
def compute_base_error(model, z, grid_size, mask_type="random",
                      base_ratio=None, base_runs=None):
    """批量计算 base 误差, base_runs 次合并为 1 次 batch forward"""
    if base_ratio is None:
        base_ratio = config.EVAL_BASE_MASK_RATIO
    if base_runs is None:
        base_runs = config.EVAL_BASE_RUNS

    masks = []
    for _ in range(base_runs):
        m, _ = sample_mask(grid_size, base_ratio, mask_type)
        masks.append(m)
    masks_batch = torch.stack(masks).to(z.device)  # [R, N]

    z_repeat = z.expand(base_runs, -1, -1, -1)
    z_hats = model.predict(z_repeat, masks_batch)

    z_flat = z.flatten(2).permute(0, 2, 1).expand(base_runs, -1, -1)
    z_hats_flat = z_hats.flatten(2).permute(0, 2, 1)

    l1_errors = []
    cos_errors = []
    for k in range(base_runs):
        m = masks_batch[k]
        if m.sum() == 0:
            continue
        z_m = z_flat[k, m]
        z_hat_m = z_hats_flat[k, m]

        l1_errors.append((z_m - z_hat_m).abs().mean().item())
        cos_sim = torch.nn.functional.cosine_similarity(
            z_m, z_hat_m, dim=-1
        ).mean().item()
        cos_errors.append(1.0 - cos_sim)

    return np.mean(l1_errors), np.mean(cos_errors)


@torch.no_grad()
def compute_s_var(z_hats_flat_list, masks_list, grid_size):
    if len(z_hats_flat_list) < 2:
        return 0.0

    N = grid_size * grid_size
    token_vars = []
    for i in range(N):
        preds = []
        for z_hat, mask in zip(z_hats_flat_list, masks_list):
            if mask[i]:
                preds.append(z_hat[0, i])
        if len(preds) >= 2:
            preds_stack = torch.stack(preds)
            var = preds_stack.var(dim=0).mean().item()
            token_vars.append(var)

    return np.mean(token_vars) if token_vars else 0.0


@torch.no_grad()
def compute_complexity_features(img_tensor):
    x = img_tensor[0]
    gray = x.mean(dim=0)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = torch.nn.functional.conv2d(g, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(g, sobel_y, padding=1)
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
    hf_energy = mag[hf_mask].pow(2).sum().item() / mag.pow(2).sum().item()

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
    if getattr(config, "CUDNN_BENCHMARK", False) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log(f"设备: {device}", log_file)

    model = load_model(exp_dir, device)
    grid_size = config.LATENT_SPATIAL
    K = args.K

    # --- 批量编码测试集 (比逐张编码快得多) ---
    ds_test = ChameleonTestDataset()
    dl_test = DataLoader(
        ds_test, batch_size=32, shuffle=False,
        num_workers=config.S1_NUM_WORKERS, pin_memory=True,
    )
    log(f"Chameleon 测试集: {len(ds_test)} 张, 批量编码中...", log_file)
    test_latents, test_labels, test_images = batch_encode(
        model, dl_test, device, desc="编码测试集"
    )

    # --- 批量编码验证集 ---
    mask_types = ["random", "block"]
    eval_ratios = config.EVAL_MASK_RATIOS

    log("计算 tau (基于验证集 base 误差, 按 mask_type 分别统计)...", log_file)
    ds_val = RealPatchDataset(patches_per_image=1, split="val")
    dl_val = DataLoader(
        ds_val, batch_size=32, shuffle=False,
        num_workers=config.S1_NUM_WORKERS, pin_memory=True,
    )
    val_latents, _, _ = batch_encode(model, dl_val, device, desc="编码验证集")

    tau_dict = {}
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

    # --- 评估 ---
    all_results = {}

    # --- 预计算: 每张测试图的 base_error 按 mask_type 算一次, 跨 ratio 复用 ---
    base_errors = {}  # {mtype: [(l1, cos), ...]}
    for mtype in mask_types:
        errs = []
        for z in tqdm(test_latents, desc=f"预计算 base[{mtype}]"):
            bl1, bcos = compute_base_error(model, z, grid_size, mask_type=mtype)
            errs.append((bl1, bcos))
        base_errors[mtype] = errs
    log("测试集 base_error 预计算完成", log_file)

    # --- 预计算: 复杂度特征 (与 mask 无关, 只算一次) ---
    complexity_feats = []
    for x in tqdm(test_images, desc="计算复杂度特征"):
        complexity_feats.append(compute_complexity_features(x))

    for mtype in mask_types:
        for mratio in eval_ratios:
            condition = f"{mtype}_{int(mratio * 100)}%"
            log(f"\n--- 评估条件: {condition}, K={K} ---", log_file)

            scores_logratio_l1 = []
            scores_logratio_cos = []
            scores_var = []
            actual_ratios_all = []

            for i in tqdm(range(len(test_latents)), desc=f"{condition}"):
                z = test_latents[i]

                mean_l1, mean_cos, z_hats, used_masks, act_ratio = compute_masked_error(
                    model, z, mratio, mtype, K, grid_size
                )

                base_l1, base_cos = base_errors[mtype][i]

                tau_l1 = tau_dict[mtype]["l1"]
                tau_cos = tau_dict[mtype]["cos"]

                s_lr_l1 = np.log((mean_l1 + tau_l1) / (base_l1 + tau_l1))
                scores_logratio_l1.append(s_lr_l1)

                s_lr_cos = np.log((mean_cos + tau_cos) / (base_cos + tau_cos))
                scores_logratio_cos.append(s_lr_cos)

                if len(z_hats) >= 2:
                    s_var = compute_s_var(z_hats, used_masks, grid_size)
                else:
                    s_var = 0.0
                scores_var.append(s_var)
                actual_ratios_all.append(act_ratio)

            labels = np.array(test_labels)
            scores_l1 = np.array(scores_logratio_l1)
            scores_cos = np.array(scores_logratio_cos)
            s_vars = np.array(scores_var)

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

            try:
                auroc_var = roc_auc_score(labels, s_vars)
                aupr_var = average_precision_score(labels, s_vars)
            except ValueError:
                auroc_var = aupr_var = float("nan")

            mean_act_ratio = np.mean(actual_ratios_all)
            log(f"  实际 mask 比例均值: {mean_act_ratio:.4f} (目标 {mratio:.2f})", log_file)
            log(f"  S_logratio(L1):  AUROC={auroc_l1:.4f}, AUPR={aupr_l1:.4f}", log_file)
            log(f"  S_logratio(cos): AUROC={auroc_cos:.4f}, AUPR={aupr_cos:.4f}", log_file)
            log(f"  S_var:           AUROC={auroc_var:.4f}, AUPR={aupr_var:.4f}", log_file)
            log(f"  S_var 均值: real={s_vars[labels == 0].mean():.6f}, "
                f"fake={s_vars[labels == 1].mean():.6f}", log_file)

            cf_keys = ["edge_density", "hf_energy", "patch_entropy", "grad_energy"]
            for key in cf_keys:
                vals = np.array([cf[key] for cf in complexity_feats[:len(test_latents)]])
                if np.std(vals) > 1e-10 and np.std(scores_l1) > 1e-10:
                    corr, pval = pearsonr(vals, scores_l1)
                    log(f"  {key} vs S_logratio(L1): r={corr:.4f}, p={pval:.4f}", log_file)

            all_results[condition] = {
                "auroc_l1": float(auroc_l1),
                "aupr_l1": float(aupr_l1),
                "auroc_cos": float(auroc_cos),
                "aupr_cos": float(aupr_cos),
                "auroc_var": float(auroc_var),
                "aupr_var": float(aupr_var),
                "s_var_real_mean": float(s_vars[labels == 0].mean()),
                "s_var_fake_mean": float(s_vars[labels == 1].mean()),
                "tau_l1": float(tau_dict[mtype]["l1"]),
                "tau_cos": float(tau_dict[mtype]["cos"]),
                "actual_mask_ratio_mean": float(mean_act_ratio),
                "K": K,
            }

    result_path = os.path.join(eval_dir, f"eval_results_K{K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent MAE Zero-Shot 评估 (5090 优化版)")
    parser.add_argument("--exp_name", type=str, default="A0",
                        help="实验名称")
    parser.add_argument("--K", type=int, default=config.EVAL_K,
                        help="每张图的探测次数")
    args = parser.parse_args()
    evaluate(args)
