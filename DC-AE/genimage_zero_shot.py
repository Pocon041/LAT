import os
import json
import math
import argparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score

import config
from dataset import _collect_images, _resize_shorter_side, RealPatchDataset
from evaluate import (
    log,
    load_model,
    compute_masked_error_channelwise,
    compute_base_error_channelwise,
    compute_s_var_channelwise,
)


DEFAULT_MASK_TYPES = ["random", "block"]


class GenImageGeneratorDataset(Dataset):
    def __init__(self, generator_dir, split="val"):
        super().__init__()
        self.generator_name = os.path.basename(generator_dir.rstrip("\\/"))
        self.items = []
        real_dir = os.path.join(generator_dir, split, "nature")
        fake_dir = os.path.join(generator_dir, split, "ai")
        for path in _collect_images(real_dir):
            self.items.append((path, 0))
        for path in _collect_images(fake_dir):
            self.items.append((path, 1))
        self.transform = transforms.Compose([
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = _resize_shorter_side(img, target_short=288)
        img = self.transform(img)
        return img, label


def list_generators(genimage_root, selected=None):
    selected_set = set(selected) if selected else None
    paths = []
    for name in sorted(os.listdir(genimage_root)):
        path = os.path.join(genimage_root, name)
        if not os.path.isdir(path):
            continue
        if not os.path.isdir(os.path.join(path, "train")):
            continue
        if not os.path.isdir(os.path.join(path, "val")):
            continue
        if selected_set is not None and name not in selected_set:
            continue
        paths.append(path)
    return paths


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def safe_ap(labels, scores):
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def search_best_acc_threshold(labels, scores, steps=400):
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
    for thr in np.linspace(lo, hi, steps):
        preds = (scores >= thr).astype(np.int64)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, float(best_acc)


@torch.no_grad()
def estimate_tau(model, device, grid_size, struct_ch, num_workers, log_file):
    ds_val = RealPatchDataset(patches_per_image=1, split="val")
    dl_val = make_loader(ds_val, batch_size=1, num_workers=num_workers)

    val_latents = []
    for batch in tqdm(dl_val, desc="提取验证集 latent", leave=False):
        x = batch.to(device, non_blocking=True)
        z = model.encode(x)
        val_latents.append(z)

    tau_dict = {}
    for mtype in DEFAULT_MASK_TYPES:
        struct_l1_list = []
        struct_cos_list = []
        detail_l1_list = []
        detail_cos_list = []
        for z in tqdm(val_latents, desc=f"tau[{mtype}]", leave=False):
            base_err = compute_base_error_channelwise(model, z, grid_size, mtype, struct_ch)
            struct_l1_list.append(base_err["struct"]["l1"])
            struct_cos_list.append(base_err["struct"]["cos"])
            detail_l1_list.append(base_err["detail"]["l1"])
            detail_cos_list.append(base_err["detail"]["cos"])
        tau_dict[mtype] = {
            "struct_l1": float(np.percentile(struct_l1_list, config.TAU_PERCENTILE)),
            "struct_cos": float(np.percentile(struct_cos_list, config.TAU_PERCENTILE)),
            "detail_l1": float(np.percentile(detail_l1_list, config.TAU_PERCENTILE)),
            "detail_cos": float(np.percentile(detail_cos_list, config.TAU_PERCENTILE)),
        }
        log(
            f"tau[{mtype}]: struct_l1={tau_dict[mtype]['struct_l1']:.6f}, detail_l1={tau_dict[mtype]['detail_l1']:.6f}",
            log_file,
        )
    return tau_dict


@torch.no_grad()
def collect_condition_scores(model, loader, device, grid_size, struct_ch, K, mask_type, mask_ratio, tau, desc=None):
    labels = []
    scores_struct = []
    scores_detail = []
    scores_couple = []
    scores_var_struct = []
    scores_var_detail = []
    scores_var_total = []
    actual_ratios = []

    iterator = loader
    if desc is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        xb, yb = batch
        xb = xb.to(device, non_blocking=True)
        for i in range(xb.shape[0]):
            x = xb[i:i + 1]
            label = int(yb[i].item())
            z = model.encode(x)
            mask_err = compute_masked_error_channelwise(model, z, mask_ratio, mask_type, K, grid_size, struct_ch)
            base_err = compute_base_error_channelwise(model, z, grid_size, mask_type, struct_ch)

            s_struct = np.log(
                (mask_err["struct"]["l1"] + tau["struct_l1"]) /
                (base_err["struct"]["l1"] + tau["struct_l1"])
            )
            s_detail = np.log(
                (mask_err["detail"]["l1"] + tau["detail_l1"]) /
                (base_err["detail"]["l1"] + tau["detail_l1"])
            )
            s_var_s, s_var_d, s_var_t = compute_s_var_channelwise(
                mask_err["z_hats"], mask_err["masks"], grid_size, struct_ch
            )
            s_couple = np.log(
                (s_var_d + tau["detail_l1"]) /
                (s_var_s + tau["struct_l1"] + 1e-10)
            )

            labels.append(label)
            scores_struct.append(float(s_struct))
            scores_detail.append(float(s_detail))
            scores_couple.append(float(s_couple))
            scores_var_struct.append(float(s_var_s))
            scores_var_detail.append(float(s_var_d))
            scores_var_total.append(float(s_var_t))
            actual_ratios.append(float(mask_err["actual_ratio"]))

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "S_struct": np.asarray(scores_struct, dtype=np.float64),
        "S_detail": np.asarray(scores_detail, dtype=np.float64),
        "S_couple_lite": np.asarray(scores_couple, dtype=np.float64),
        "S_var_struct": np.asarray(scores_var_struct, dtype=np.float64),
        "S_var_detail": np.asarray(scores_var_detail, dtype=np.float64),
        "S_var_total": np.asarray(scores_var_total, dtype=np.float64),
        "actual_ratios": np.asarray(actual_ratios, dtype=np.float64),
    }


def summarize_metrics(calib_data, test_data):
    labels_calib = calib_data["labels"]
    labels_test = test_data["labels"]
    metrics = {}
    for score_name in [
        "S_struct",
        "S_detail",
        "S_couple_lite",
        "S_var_struct",
        "S_var_detail",
        "S_var_total",
    ]:
        thr, calib_acc = search_best_acc_threshold(labels_calib, calib_data[score_name])
        ap = safe_ap(labels_test, test_data[score_name])
        if math.isnan(thr):
            acc = float("nan")
        else:
            preds = (test_data[score_name] >= thr).astype(np.int64)
            acc = float(accuracy_score(labels_test, preds))
        metrics[score_name] = {
            "threshold": float(thr),
            "calib_acc": float(calib_acc),
            "test_ap": float(ap),
            "test_acc": float(acc),
        }
    return metrics


def evaluate_genimage(args):
    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, f"genimage_zero_shot_{args.test_split}_K{args.K}.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}", log_file)

    model = load_model(exp_dir, device)
    grid_size = config.LATENT_SPATIAL
    struct_ch = config.STRUCT_CHANNELS

    mask_types = args.mask_types if args.mask_types else DEFAULT_MASK_TYPES
    mask_ratios = args.mask_ratios if args.mask_ratios else config.EVAL_MASK_RATIOS
    generator_paths = list_generators(args.genimage_root, selected=args.generators)
    if len(generator_paths) == 0:
        raise RuntimeError(f"在 {args.genimage_root} 中没有找到可用生成器目录")

    log("开始估计 tau ...", log_file)
    tau_dict = estimate_tau(model, device, grid_size, struct_ch, args.num_workers, log_file)

    results = {
        "meta": {
            "exp_name": args.exp_name,
            "genimage_root": args.genimage_root,
            "calib_split": args.calib_split,
            "test_split": args.test_split,
            "K": args.K,
            "mask_types": mask_types,
            "mask_ratios": mask_ratios,
            "tau": tau_dict,
        }
    }

    for generator_path in generator_paths:
        generator_name = os.path.basename(generator_path)
        ds_calib = GenImageGeneratorDataset(generator_path, split=args.calib_split)
        ds_test = GenImageGeneratorDataset(generator_path, split=args.test_split)
        dl_calib = make_loader(ds_calib, args.batch_size, args.num_workers)
        dl_test = make_loader(ds_test, args.batch_size, args.num_workers)

        log(
            f"\n生成器 {generator_name}: calib={len(ds_calib)}, test={len(ds_test)}",
            log_file,
        )

        results[generator_name] = {
            "meta": {
                "calib_size": len(ds_calib),
                "test_size": len(ds_test),
                "real_test": int(sum(label == 0 for _, label in ds_test.items)),
                "fake_test": int(sum(label == 1 for _, label in ds_test.items)),
            }
        }

        for mask_type in mask_types:
            tau = tau_dict[mask_type]
            for mask_ratio in mask_ratios:
                condition = f"{mask_type}_{int(mask_ratio * 100)}%"
                log(f"  条件 {condition}", log_file)
                calib_data = collect_condition_scores(
                    model,
                    dl_calib,
                    device,
                    grid_size,
                    struct_ch,
                    args.K,
                    mask_type,
                    mask_ratio,
                    tau,
                    desc=f"{generator_name} calib {condition}",
                )
                test_data = collect_condition_scores(
                    model,
                    dl_test,
                    device,
                    grid_size,
                    struct_ch,
                    args.K,
                    mask_type,
                    mask_ratio,
                    tau,
                    desc=f"{generator_name} test {condition}",
                )
                metrics = summarize_metrics(calib_data, test_data)
                metrics["actual_mask_ratio_calib_mean"] = float(np.mean(calib_data["actual_ratios"])) if len(calib_data["actual_ratios"]) > 0 else 0.0
                metrics["actual_mask_ratio_test_mean"] = float(np.mean(test_data["actual_ratios"])) if len(test_data["actual_ratios"]) > 0 else 0.0
                results[generator_name][condition] = metrics
                log(
                    "    " + " | ".join(
                        f"{score}: AP={item['test_ap']:.4f}, ACC={item['test_acc']:.4f}"
                        for score, item in metrics.items()
                        if isinstance(item, dict) and "test_ap" in item
                    ),
                    log_file,
                )

    out_path = os.path.join(eval_dir, f"genimage_zero_shot_{args.test_split}_K{args.K}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"结果已保存到: {out_path}", log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DC-AE 在 Genimage-Tiny 上的 zero-shot 分生成器评测")
    parser.add_argument("--exp_name", type=str, default="Exp2_A0")
    parser.add_argument("--genimage_root", type=str, default=r"D:\Genimage-Tiny")
    parser.add_argument("--calib_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--test_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--K", type=int, default=config.EVAL_K)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mask_types", nargs="*", default=None)
    parser.add_argument("--mask_ratios", nargs="*", type=float, default=None)
    parser.add_argument("--generators", nargs="*", default=None)
    args = parser.parse_args()
    evaluate_genimage(args)
