"""
评估脚本: 对指定模型跑 Exp0 + Exp1(f3_pca_truncate) 在源域和 Chameleon 上

用法:
  python evaluate.py --mode A0
  python evaluate.py --mode A1
  python evaluate.py --mode all
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import config
from model import SimpleAE
from dataset import RealTrainDataset, RealValDataset, GenImageTestDataset, ChameleonTestDataset
from losses import load_lpips_vgg, FreqLoss


# ==================== PCA 工具 ====================

def fit_pca(model, dataloader, device, n_samples=2000):
    """在 real 数据上拟合 PCA"""
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="拟合PCA", file=sys.stdout):
            x = batch.to(device)
            z = model.encode(x)
            all_z.append(z.view(z.size(0), -1).cpu())
            if sum(t.size(0) for t in all_z) >= n_samples:
                break
    all_z = torch.cat(all_z, dim=0)[:n_samples]
    mean = all_z.mean(dim=0)
    centered = all_z - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    return mean, Vt


def f3_pca_truncate(z, pca_mean, pca_Vt, n_components, latent_shape):
    B = z.size(0)
    z_flat = z.view(B, -1)
    centered = z_flat - pca_mean
    proj = centered @ pca_Vt[:n_components].T @ pca_Vt[:n_components]
    return (proj + pca_mean).view(B, *latent_shape)


# ==================== 预计算缓存 ====================

def precompute_cache(model, dataloader, lpips_fn, device):
    all_z = []
    all_d_base = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for img, label, sources in tqdm(dataloader, desc="预计算缓存", file=sys.stdout):
            img = img.to(device)
            z = model.encode(img)
            recon = model.decode(z)
            d_base = lpips_fn(img * 2 - 1, recon * 2 - 1).view(-1)
            all_z.append(z.cpu().half())
            all_d_base.append(d_base.cpu())
            all_labels.append(label)
    return torch.cat(all_z), torch.cat(all_d_base), torch.cat(all_labels)


def evaluate_f3_fast(model, val_loader, all_z, all_d_base, all_labels,
                     pca_mean, pca_Vt, n_components, latent_shape,
                     lpips_fn, device):
    deltas = torch.zeros(all_z.size(0))
    idx = 0
    model.eval()
    with torch.no_grad():
        for img, label, sources in tqdm(val_loader, desc=f"    f3 n={n_components}", leave=False, file=sys.stdout):
            B = img.size(0)
            img = img.to(device)
            z = all_z[idx:idx+B].to(device).float()
            z_f = f3_pca_truncate(z, pca_mean, pca_Vt, n_components, latent_shape)
            recon_f = model.decode(z_f)
            d_f = lpips_fn(img * 2 - 1, recon_f * 2 - 1).view(-1).cpu()
            d_base = all_d_base[idx:idx+B]
            deltas[idx:idx+B] = d_f - d_base
            idx += B

    labels_np = all_labels.numpy()
    dr = deltas[labels_np == 0].tolist()
    df = deltas[labels_np == 1].tolist()
    return dr, df


# ==================== Exp0 ====================

def run_exp0(model, dataloader, lpips_fn, freq_fn, device, name):
    print(f"\n{'='*70}", flush=True)
    print(f"Exp0 [{name}]: Baseline 重建误差", flush=True)
    print(f"{'='*70}", flush=True)

    all_real = {"l1": [], "lpips": [], "freq": []}
    all_fake = {"l1": [], "lpips": [], "freq": []}

    model.eval()
    with torch.no_grad():
        for img, label, sources in tqdm(dataloader, desc="Exp0", file=sys.stdout):
            img = img.to(device)
            recon, z = model(img)
            for i in range(img.size(0)):
                x_i = img[i:i+1]
                r_i = recon[i:i+1]
                target = all_fake if label[i].item() == 1 else all_real
                target["l1"].append((x_i - r_i).abs().mean().item())
                target["lpips"].append(lpips_fn(x_i * 2 - 1, r_i * 2 - 1).item())
                target["freq"].append(freq_fn(x_i, r_i).item())

    result = {}
    print(f"\n{'Split':<8} {'L1':>8} {'LPIPS':>8} {'Freq':>8} {'N':>6}")
    print("-" * 45)
    for split, agg in [("real", all_real), ("fake", all_fake)]:
        if not agg["l1"]:
            continue
        result[split] = {k: float(np.mean(v)) for k, v in agg.items()}
        result[split]["n"] = len(agg["l1"])
        print(f"{split:<8} {np.mean(agg['l1']):8.4f} {np.mean(agg['lpips']):8.4f} "
              f"{np.mean(agg['freq']):8.4f} {len(agg['l1']):6d}")

    if all_real["l1"] and all_fake["l1"]:
        print("\nOOD Ratio (fake/real):")
        for k in ["l1", "lpips", "freq"]:
            ratio = np.mean(all_fake[k]) / (np.mean(all_real[k]) + 1e-8)
            print(f"  {k:>6}: {ratio:.3f}")
        result["ood_ratio"] = {k: np.mean(all_fake[k]) / (np.mean(all_real[k]) + 1e-8) for k in ["l1", "lpips", "freq"]}

    return result


# ==================== Exp1 f3 ====================

def run_exp1_f3(model, val_loader, calib_loader, lpips_fn, device, name):
    print(f"\n{'='*70}", flush=True)
    print(f"Exp1 f3_pca [{name}]: Latent PCA 截断", flush=True)
    print(f"{'='*70}", flush=True)

    # 预计算
    all_z, all_d_base, all_labels = precompute_cache(model, val_loader, lpips_fn, device)
    print(f"缓存: {all_z.shape[0]} 样本", flush=True)

    # PCA 拟合 (用 calib real)
    pca_mean, pca_Vt = fit_pca(model, calib_loader, device, n_samples=2000)
    pca_mean = pca_mean.to(device)
    pca_Vt = pca_Vt.to(device)
    max_rank = pca_Vt.size(0)
    latent_shape = (config.LATENT_CHANNELS, config.LATENT_SPATIAL, config.LATENT_SPATIAL)
    print(f"PCA max_rank={max_rank}", flush=True)

    keep_ratios = [0.9, 0.7, 0.5, 0.3, 0.1]
    results = {}

    for kr in keep_ratios:
        n_pca = max(1, int(max_rank * kr))
        print(f"  keep={kr} (n_pca={n_pca})...", flush=True)
        dr, df = evaluate_f3_fast(model, val_loader, all_z, all_d_base, all_labels,
                                  pca_mean, pca_Vt, n_pca, latent_shape, lpips_fn, device)

        labels = [0] * len(dr) + [1] * len(df)
        scores = dr + df
        auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0

        results[f"kr{kr}"] = {
            "keep_ratio": kr, "n_pca": n_pca,
            "real_delta": float(np.mean(dr)), "fake_delta": float(np.mean(df)),
            "auroc": auroc,
        }
        print(f"    real={np.mean(dr):.4f}, fake={np.mean(df):.4f}, AUROC={auroc:.4f}", flush=True)

    return results


# ==================== Main ====================

def evaluate_one_model(mode, device, lpips_fn, freq_fn):
    exp_dir = os.path.join(config.EXP_ROOT, mode)
    ckpt = os.path.join(exp_dir, "best_ae.pth")
    if not os.path.exists(ckpt):
        print(f"跳过 {mode}: {ckpt} 不存在", flush=True)
        return

    print(f"\n{'#'*70}", flush=True)
    print(f"# 评估模型: {mode}", flush=True)
    print(f"{'#'*70}", flush=True)

    model = SimpleAE(latent_channels=config.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"加载: {ckpt}", flush=True)

    # calib: real val (DIV2K val)
    calib_ds = RealValDataset()
    calib_loader = DataLoader(calib_ds, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    all_results = {}

    # --- 源域 (GenImage) ---
    gen_ds = GenImageTestDataset()
    gen_loader = DataLoader(gen_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)
    n_real = sum(1 for _, l, _ in gen_ds.items if l == 0)
    n_fake = sum(1 for _, l, _ in gen_ds.items if l == 1)
    print(f"\nGenImage: real={n_real}, fake={n_fake}", flush=True)

    all_results["genimage_exp0"] = run_exp0(model, gen_loader, lpips_fn, freq_fn, device, "GenImage")
    all_results["genimage_exp1_f3"] = run_exp1_f3(model, gen_loader, calib_loader, lpips_fn, device, "GenImage")

    # --- Chameleon zero-shot ---
    cham_ds = ChameleonTestDataset(max_per_class=2000)
    cham_loader = DataLoader(cham_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)
    n_real = sum(1 for _, l, _ in cham_ds.items if l == 0)
    n_fake = sum(1 for _, l, _ in cham_ds.items if l == 1)
    print(f"\nChameleon: real={n_real}, fake={n_fake}", flush=True)

    all_results["chameleon_exp0"] = run_exp0(model, cham_loader, lpips_fn, freq_fn, device, "Chameleon")
    all_results["chameleon_exp1_f3"] = run_exp1_f3(model, cham_loader, calib_loader, lpips_fn, device, "Chameleon")

    # 保存
    save_path = os.path.join(exp_dir, "eval_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到 {save_path}", flush=True)

    return all_results


def print_comparison(all_model_results):
    """打印多个模型的 f3 AUROC 对比表"""
    print(f"\n{'='*90}", flush=True)
    print("模型对比: f3_pca_truncate AUROC", flush=True)
    print(f"{'='*90}", flush=True)

    keep_ratios = ["kr0.9", "kr0.7", "kr0.5", "kr0.3", "kr0.1"]

    # GenImage
    print(f"\n--- GenImage (源域) ---")
    header = f"{'Mode':<8}"
    for kr in keep_ratios:
        header += f" {kr:>8}"
    print(header)
    print("-" * 55)
    for mode, results in all_model_results.items():
        if results is None or "genimage_exp1_f3" not in results:
            continue
        row = f"{mode:<8}"
        for kr in keep_ratios:
            auroc = results["genimage_exp1_f3"].get(kr, {}).get("auroc", 0)
            row += f" {auroc:8.4f}"
        print(row)

    # Chameleon
    print(f"\n--- Chameleon (zero-shot) ---")
    print(header)
    print("-" * 55)
    for mode, results in all_model_results.items():
        if results is None or "chameleon_exp1_f3" not in results:
            continue
        row = f"{mode:<8}"
        for kr in keep_ratios:
            auroc = results["chameleon_exp1_f3"].get(kr, {}).get("auroc", 0)
            row += f" {auroc:8.4f}"
        print(row)

    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="A0, A1, A2, A3, or 'all'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = load_lpips_vgg(device)
    freq_fn = FreqLoss().to(device)

    if args.mode == "all":
        modes = ["A0", "A1", "A2", "A3"]
    else:
        modes = [args.mode]

    all_model_results = {}
    for mode in modes:
        result = evaluate_one_model(mode, device, lpips_fn, freq_fn)
        all_model_results[mode] = result

    if len(all_model_results) > 1:
        print_comparison(all_model_results)


if __name__ == "__main__":
    main()
