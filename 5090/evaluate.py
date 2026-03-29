import os
import json
import math
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import config
from dataset import ChameleonTestDataset, RealPatchDataset, FolderBinaryDataset
from model import PixelMAE, sample_mask, patch_mask_to_pixel_mask, erode_pixel_mask
from losses import masked_l1_mean, masked_laplacian_l1_mean


def log(msg, log_file=None):
    print(msg)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_backend():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    if config.ENABLE_CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    if config.ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def move_to_device(x, device):
    x = x.to(device, non_blocking=True)
    if config.USE_CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def build_model(device):
    model_raw = PixelMAE().to(device)
    if config.USE_CHANNELS_LAST:
        model_raw = model_raw.to(memory_format=torch.channels_last)
    model = model_raw
    if config.ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model_raw)
    return model_raw, model


def make_loader(dataset, batch_size, shuffle=False, num_workers=None):
    if num_workers is None:
        num_workers = config.EVAL_NUM_WORKERS
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.PIN_MEMORY,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = config.PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = config.PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def load_model(exp_dir, device):
    model_raw, model = build_model(device)
    best_path = os.path.join(exp_dir, "best_model.pth")
    last_path = os.path.join(exp_dir, "last_model.pth")
    ckpt_path = best_path if os.path.exists(best_path) else last_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型权重: {best_path} 或 {last_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_raw.load_state_dict(ckpt["model"])
    model.eval()
    return model_raw, model, ckpt_path, ckpt.get("epoch", -1)


def safe_auc_ap(labels, scores):
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float("nan")
    try:
        ap = average_precision_score(labels, scores)
    except Exception:
        ap = float("nan")
    return float(auroc), float(ap)


def compute_complexity_features(x):
    gray = x[0].mean(dim=0)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    g = gray.unsqueeze(0).unsqueeze(0)
    gx = torch.nn.functional.conv2d(g, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(g, sobel_y, padding=1)
    grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()
    edge_density = grad_mag.mean().item()
    grad_energy = (gx.pow(2) + gy.pow(2)).mean().item()

    fft = torch.fft.rfft2(gray, norm="ortho")
    mag = fft.abs()
    h, w2 = mag.shape
    fy = torch.arange(h, device=x.device, dtype=x.dtype)
    fx = torch.arange(w2, device=x.device, dtype=x.dtype)
    fy[fy > h // 2] -= h
    gy_freq, gx_freq = torch.meshgrid(fy, fx, indexing="ij")
    radius = (gy_freq.pow(2) + gx_freq.pow(2)).sqrt()
    max_r = radius.max().clamp_min(1e-8)
    hf_mask = radius > 0.5 * max_r
    total_energy = mag.pow(2).sum().clamp_min(1e-8)
    hf_energy = mag[hf_mask].pow(2).sum().item() / total_energy.item()

    patch_size = 32
    entropies = []
    for i in range(0, gray.shape[0] - patch_size + 1, patch_size):
        for j in range(0, gray.shape[1] - patch_size + 1, patch_size):
            patch = gray[i:i + patch_size, j:j + patch_size].flatten()
            hist = torch.histc(patch, bins=32, min=0.0, max=1.0)
            hist = hist / hist.sum().clamp_min(1e-8)
            hist = hist[hist > 0]
            entropies.append((-(hist * hist.log())).sum().item())
    patch_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "edge_density": float(edge_density),
        "hf_energy": float(hf_energy),
        "patch_entropy": float(patch_entropy),
        "grad_energy": float(grad_energy),
    }


def sample_valid_mask(mask_type, mask_ratio, device):
    for _ in range(config.MAX_MASK_SAMPLE_TRIES):
        mask, actual_ratio = sample_mask(config.GRID_SIZE, mask_ratio, mask_type)
        mask = mask.to(device)
        pixel_mask = patch_mask_to_pixel_mask(mask, config.IMG_SIZE, config.PATCH_SIZE)
        core_mask = erode_pixel_mask(pixel_mask.unsqueeze(0), config.CORE_EROSION_PX)[0]
        core_pixels = int(core_mask.sum().item())
        if core_pixels >= config.CORE_MIN_PIXELS:
            return {
                "mask": mask,
                "actual_ratio": float(actual_ratio),
                "pixel_mask": pixel_mask,
                "core_mask": core_mask,
                "core_pixels": core_pixels,
            }
    raise RuntimeError(
        f"无法在 {config.MAX_MASK_SAMPLE_TRIES} 次采样内获得有效 masked core: "
        f"type={mask_type}, ratio={mask_ratio}"
    )


def batch_sample_valid_masks(mask_type, mask_ratio, device, count):
    masks = []
    actual_ratios = []
    core_masks = []
    core_pixels_list = []
    for _ in range(count):
        info = sample_valid_mask(mask_type, mask_ratio, device)
        masks.append(info["mask"])
        actual_ratios.append(info["actual_ratio"])
        core_masks.append(info["core_mask"])
        core_pixels_list.append(info["core_pixels"])
    return {
        "masks": torch.stack(masks, dim=0),
        "actual_ratios": actual_ratios,
        "core_masks": torch.stack(core_masks, dim=0),
        "core_pixels": core_pixels_list,
    }


@torch.no_grad()
def run_single_prediction(model, x, mask_info):
    mask = mask_info["mask"].unsqueeze(0)
    recon, pred_full, _, _ = model.reconstruct(x, mask, copy_back=True)
    core_mask = mask_info["core_mask"].unsqueeze(0)
    raw_err = masked_l1_mean(recon, x, core_mask).item()
    hf_err = masked_laplacian_l1_mean(recon, x, core_mask).item()
    return {
        "recon": recon,
        "pred_full": pred_full,
        "raw_err": float(raw_err),
        "hf_err": float(hf_err),
        "core_mask": core_mask,
        "pixel_mask": mask_info["pixel_mask"].unsqueeze(0),
        "actual_ratio": mask_info["actual_ratio"],
        "core_pixels": mask_info["core_pixels"],
    }


@torch.no_grad()
def run_batch_prediction(model, x, batch_mask_info):
    masks = batch_mask_info["masks"]
    K = masks.shape[0]
    x_repeat = x.expand(K, -1, -1, -1)
    recon, pred_full, pred_patches, gt_patches = model.reconstruct(x_repeat, masks, copy_back=True)
    core_masks = batch_mask_info["core_masks"]
    raw_errs = []
    hf_errs = []
    for i in range(K):
        raw_err = masked_l1_mean(recon[i:i+1], x, core_masks[i:i+1]).item()
        hf_err = masked_laplacian_l1_mean(recon[i:i+1], x, core_masks[i:i+1]).item()
        raw_errs.append(raw_err)
        hf_errs.append(hf_err)
    return {
        "recons": recon,
        "raw_errs": raw_errs,
        "hf_errs": hf_errs,
        "core_masks": core_masks,
        "actual_ratios": batch_mask_info["actual_ratios"],
        "core_pixels": batch_mask_info["core_pixels"],
    }


@torch.no_grad()
def compute_base_errors(model, x, mask_type):
    batch_info = batch_sample_valid_masks(mask_type, config.EVAL_BASE_MASK_RATIO, x.device, config.EVAL_BASE_RUNS)
    out = run_batch_prediction(model, x, batch_info)
    return {
        "raw": float(np.mean(out["raw_errs"])),
        "hf": float(np.mean(out["hf_errs"])),
    }


@torch.no_grad()
def compute_pixel_var(recon_list, core_mask_list):
    if len(recon_list) < 2:
        return 0.0
    x_stack = torch.stack([r[0] for r in recon_list], dim=0)
    m_stack = torch.stack([m[0] for m in core_mask_list], dim=0).float()
    m_stack = m_stack.expand(-1, x_stack.shape[1], -1, -1)
    count = m_stack.sum(dim=0)
    valid = count >= 2.0
    if not valid.any():
        return 0.0
    sum_x = (x_stack * m_stack).sum(dim=0)
    sum_x2 = (x_stack.pow(2) * m_stack).sum(dim=0)
    mean = sum_x / count.clamp_min(1.0)
    var = sum_x2 / count.clamp_min(1.0) - mean.pow(2)
    score = var[valid].mean().item()
    return float(score)


@torch.no_grad()
def compute_pixel_var_batch(recons, core_masks):
    if recons.shape[0] < 2:
        return 0.0
    m_stack = core_masks.expand(-1, recons.shape[1], -1, -1).float()
    count = m_stack.sum(dim=0)
    valid = count >= 2.0
    if not valid.any():
        return 0.0
    sum_x = (recons * m_stack).sum(dim=0)
    sum_x2 = (recons.pow(2) * m_stack).sum(dim=0)
    mean = sum_x / count.clamp_min(1.0)
    var = sum_x2 / count.clamp_min(1.0) - mean.pow(2)
    return float(var[valid].mean().item())


@torch.no_grad()
def compute_condition_scores(model, x, mask_type, mask_ratio, K):
    base = compute_base_errors(model, x, mask_type)
    batch_info = batch_sample_valid_masks(mask_type, mask_ratio, x.device, K)
    out = run_batch_prediction(model, x, batch_info)
    pixel_var = compute_pixel_var_batch(out["recons"], out["core_masks"])
    return {
        "base_raw": base["raw"],
        "base_hf": base["hf"],
        "mean_raw": float(np.mean(out["raw_errs"])),
        "mean_hf": float(np.mean(out["hf_errs"])),
        "pixel_var": float(pixel_var),
        "actual_ratio": float(np.mean(out["actual_ratios"])),
        "core_pixels": float(np.mean(out["core_pixels"])),
    }


def search_best_acc_threshold(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    lo = float(scores.min())
    hi = float(scores.max())
    if math.isclose(lo, hi):
        preds = (scores >= lo).astype(np.int64)
        return float(lo), float(accuracy_score(labels, preds))
    best_thr = lo
    best_acc = -1.0
    for thr in np.linspace(lo, hi, config.THRESHOLD_SEARCH_STEPS):
        preds = (scores >= thr).astype(np.int64)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, float(best_acc)


def try_build_calibration_loader():
    if config.CALIB_REAL_DIR is None or config.CALIB_FAKE_DIR is None:
        return None
    if not os.path.isdir(config.CALIB_REAL_DIR) or not os.path.isdir(config.CALIB_FAKE_DIR):
        return None
    ds = FolderBinaryDataset(
        config.CALIB_REAL_DIR,
        config.CALIB_FAKE_DIR,
        preload=False,
        desc="calib",
    )
    if len(ds) == 0:
        return None
    return make_loader(ds, config.EVAL_BATCH_SIZE, shuffle=False, num_workers=config.EVAL_NUM_WORKERS)


@torch.no_grad()
def collect_scores_for_loader(model, loader, mask_type, mask_ratio, tau_dict, K, device, log_prefix=None):
    scores_raw = []
    scores_hf = []
    scores_var = []
    labels = []
    complexity = []
    actual_ratios = []
    core_pixels = []
    iterator = loader
    if log_prefix is not None:
        iterator = tqdm(loader, desc=log_prefix, leave=False)
    for batch in iterator:
        xb, yb = batch[:2]
        xb = move_to_device(xb, device)
        for i in range(xb.shape[0]):
            x = xb[i:i + 1]
            label = int(yb[i].item())
            score_item = compute_condition_scores(model, x, mask_type, mask_ratio, K)
            tau = tau_dict[mask_type]
            s_raw = math.log((score_item["mean_raw"] + tau["raw"]) / (score_item["base_raw"] + tau["raw"]))
            s_hf = math.log((score_item["mean_hf"] + tau["hf"]) / (score_item["base_hf"] + tau["hf"]))
            scores_raw.append(s_raw)
            scores_hf.append(s_hf)
            scores_var.append(score_item["pixel_var"])
            labels.append(label)
            actual_ratios.append(score_item["actual_ratio"])
            core_pixels.append(score_item["core_pixels"])
            complexity.append(compute_complexity_features(x))
    return {
        "labels": np.array(labels),
        "scores_raw": np.array(scores_raw, dtype=np.float64),
        "scores_hf": np.array(scores_hf, dtype=np.float64),
        "scores_var": np.array(scores_var, dtype=np.float64),
        "actual_ratios": np.array(actual_ratios, dtype=np.float64),
        "core_pixels": np.array(core_pixels, dtype=np.float64),
        "complexity": complexity,
    }


def get_corr_map(complexity_feats, score_map):
    results = {}
    for ckey in ["edge_density", "hf_energy", "patch_entropy", "grad_energy"]:
        vals = np.array([item[ckey] for item in complexity_feats], dtype=np.float64)
        if np.std(vals) < 1e-10:
            continue
        for sname, sarr in score_map.items():
            if np.std(sarr) < 1e-10:
                continue
            corr, pval = pearsonr(vals, sarr)
            results[f"{ckey}__{sname}"] = {
                "r": float(corr),
                "p": float(pval),
            }
    return results


@torch.no_grad()
def estimate_tau(model, dl_val, device, log_file):
    tau_dict = {}
    val_items = []
    for batch in tqdm(dl_val, desc="缓存验证集", leave=False):
        x = batch
        for i in range(x.shape[0]):
            val_items.append(x[i:i + 1].clone())
    for mask_type in config.EVAL_MASK_TYPES:
        raw_list = []
        hf_list = []
        for x_cpu in tqdm(val_items, desc=f"tau[{mask_type}]", leave=False):
            x = move_to_device(x_cpu, device)
            base = compute_base_errors(model, x, mask_type)
            raw_list.append(base["raw"])
            hf_list.append(base["hf"])
        tau_raw = float(np.percentile(raw_list, config.TAU_PERCENTILE)) if raw_list else 0.0
        tau_hf = float(np.percentile(hf_list, config.TAU_PERCENTILE)) if hf_list else 0.0
        tau_dict[mask_type] = {"raw": tau_raw, "hf": tau_hf}
        log(
            f"tau[{mask_type}]: raw={tau_raw:.6f}, hf={tau_hf:.6f} ({config.TAU_PERCENTILE}% 分位数)",
            log_file,
        )
    return tau_dict


def evaluate(args):
    set_seed(config.SEED)
    configure_backend()

    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, f"eval_log_K{args.K}.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}", log_file)
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}", log_file)

    model_raw, model, ckpt_path, epoch = load_model(exp_dir, device)
    log(f"加载模型: {ckpt_path} (epoch={epoch})", log_file)

    ds_val = RealPatchDataset(split="val", patches_per_image=1)
    dl_val = make_loader(ds_val, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=config.VAL_NUM_WORKERS)
    ds_test = ChameleonTestDataset(preload=config.TEST_PRELOAD_TO_RAM)
    dl_test = make_loader(ds_test, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, num_workers=config.EVAL_NUM_WORKERS)
    dl_calib = try_build_calibration_loader()

    log(f"验证集图像数: {len(ds_val.paths)}", log_file)
    log(f"测试集图像数: {len(ds_test)}", log_file)

    log("开始估计 tau_r / tau_h ...", log_file)
    tau_dict = estimate_tau(model, dl_val, device, log_file)

    thresholds = {}
    if dl_calib is not None:
        log(f"使用独立 calibration 集: {len(dl_calib.dataset)} 张", log_file)
        log("开始在 calibration 集上搜索 ACC 阈值 ...", log_file)
        for mask_type in config.EVAL_MASK_TYPES:
            thresholds[mask_type] = {}
            for mask_ratio in config.EVAL_MASK_RATIOS:
                condition = f"{mask_type}_{int(mask_ratio * 100)}%"
                calib_scores = collect_scores_for_loader(
                    model, dl_calib, mask_type, mask_ratio, tau_dict, args.K, device,
                    log_prefix=f"calib {condition}",
                )
                thr_raw, acc_raw = search_best_acc_threshold(calib_scores["labels"], calib_scores["scores_raw"])
                thr_hf, acc_hf = search_best_acc_threshold(calib_scores["labels"], calib_scores["scores_hf"])
                thr_var, acc_var = search_best_acc_threshold(calib_scores["labels"], calib_scores["scores_var"])
                thresholds[mask_type][str(mask_ratio)] = {
                    "raw": {"threshold": thr_raw, "calib_acc": acc_raw},
                    "hf": {"threshold": thr_hf, "calib_acc": acc_hf},
                    "var": {"threshold": thr_var, "calib_acc": acc_var},
                }
                log(
                    f"calib {condition}: raw_thr={thr_raw:.6f}, hf_thr={thr_hf:.6f}, var_thr={thr_var:.6f}",
                    log_file,
                )
    else:
        log("使用固定阈值 0.0 计算 ACC (分数为 log ratio，>0 判假，<0 判真)", log_file)
        for mask_type in config.EVAL_MASK_TYPES:
            thresholds[mask_type] = {}
            for mask_ratio in config.EVAL_MASK_RATIOS:
                thresholds[mask_type][str(mask_ratio)] = {
                    "raw": {"threshold": 0.0, "calib_acc": None},
                    "hf": {"threshold": 0.0, "calib_acc": None},
                    "var": {"threshold": 0.0, "calib_acc": None},
                }

    all_results = {
        "meta": {
            "exp_name": args.exp_name,
            "K": args.K,
            "tau": tau_dict,
            "thresholds": thresholds,
        }
    }

    for mask_type in config.EVAL_MASK_TYPES:
        for mask_ratio in config.EVAL_MASK_RATIOS:
            condition = f"{mask_type}_{int(mask_ratio * 100)}%"
            log(f"\n开始评估: {condition}, K={args.K}", log_file)
            data = collect_scores_for_loader(
                model, dl_test, mask_type, mask_ratio, tau_dict, args.K, device,
                log_prefix=condition,
            )
            labels = data["labels"]
            scores_raw = data["scores_raw"]
            scores_hf = data["scores_hf"]
            scores_var = data["scores_var"]

            auroc_raw, ap_raw = safe_auc_ap(labels, scores_raw)
            auroc_hf, ap_hf = safe_auc_ap(labels, scores_hf)
            auroc_var, ap_var = safe_auc_ap(labels, scores_var)

            acc_raw = float("nan")
            acc_hf = float("nan")
            acc_var = float("nan")
            if thresholds:
                thr_info = thresholds.get(mask_type, {}).get(str(mask_ratio), None)
                if thr_info is not None:
                    acc_raw = float(accuracy_score(labels, (scores_raw >= thr_info["raw"]["threshold"]).astype(np.int64)))
                    acc_hf = float(accuracy_score(labels, (scores_hf >= thr_info["hf"]["threshold"]).astype(np.int64)))
                    acc_var = float(accuracy_score(labels, (scores_var >= thr_info["var"]["threshold"]).astype(np.int64)))

            corr_map = get_corr_map(
                data["complexity"],
                {
                    "S_raw_pixel": scores_raw,
                    "S_high_freq": scores_hf,
                    "S_pixel_var": scores_var,
                },
            )

            mean_ratio = float(np.mean(data["actual_ratios"])) if len(data["actual_ratios"]) > 0 else 0.0
            mean_core_pixels = float(np.mean(data["core_pixels"])) if len(data["core_pixels"]) > 0 else 0.0

            log(f"  实际 mask 比例: {mean_ratio:.4f}", log_file)
            log(f"  masked core 像素均值: {mean_core_pixels:.2f}", log_file)
            log(f"  S_raw-pixel: AUROC={auroc_raw:.4f}, AP={ap_raw:.4f}, ACC={acc_raw:.4f}", log_file)
            log(f"  S_high-freq: AUROC={auroc_hf:.4f}, AP={ap_hf:.4f}, ACC={acc_hf:.4f}", log_file)
            log(f"  S_pixel-var: AUROC={auroc_var:.4f}, AP={ap_var:.4f}, ACC={acc_var:.4f}", log_file)

            for key, item in corr_map.items():
                log(f"  corr {key}: r={item['r']:.4f}, p={item['p']:.4f}", log_file)

            all_results[condition] = {
                "auroc_raw_pixel": auroc_raw,
                "ap_raw_pixel": ap_raw,
                "acc_raw_pixel": acc_raw,
                "auroc_high_freq": auroc_hf,
                "ap_high_freq": ap_hf,
                "acc_high_freq": acc_hf,
                "auroc_pixel_var": auroc_var,
                "ap_pixel_var": ap_var,
                "acc_pixel_var": acc_var,
                "raw_real_mean": float(scores_raw[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
                "raw_fake_mean": float(scores_raw[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
                "hf_real_mean": float(scores_hf[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
                "hf_fake_mean": float(scores_hf[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
                "var_real_mean": float(scores_var[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
                "var_fake_mean": float(scores_var[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
                "tau_raw": float(tau_dict[mask_type]["raw"]),
                "tau_hf": float(tau_dict[mask_type]["hf"]),
                "actual_mask_ratio_mean": mean_ratio,
                "masked_core_pixels_mean": mean_core_pixels,
                "complexity_correlation": corr_map,
            }

    result_path = os.path.join(eval_dir, f"eval_results_K{args.K}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log(f"\n结果已保存到: {result_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验三 Pixel-space MAE 评估 (5090版)")
    parser.add_argument("--exp_name", type=str, default="Exp3_5090_A0")
    parser.add_argument("--K", type=int, default=config.EVAL_K)
    args = parser.parse_args()
    evaluate(args)
