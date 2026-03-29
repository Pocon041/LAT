import os
import json
import math
import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score

import config
from dataset import FolderBinaryDataset, RealPatchDataset
from evaluate import (
    log,
    set_seed,
    configure_backend,
    load_model,
    make_loader,
    estimate_tau,
    move_to_device,
    compute_condition_scores,
    search_best_acc_threshold,
)


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


def build_generator_dataset(generator_dir, split, preload):
    return FolderBinaryDataset(
        os.path.join(generator_dir, split, "nature"),
        os.path.join(generator_dir, split, "ai"),
        preload=preload,
        desc=f"{os.path.basename(generator_dir)}_{split}",
    )


def safe_ap(labels, scores):
    try:
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


@torch.no_grad()
def collect_condition_scores_for_loader(model, loader, mask_type, mask_ratio, tau_dict, K, device, desc=None):
    labels = []
    scores_raw = []
    scores_hf = []
    scores_var = []
    actual_ratios = []
    core_pixels = []

    iterator = loader
    if desc is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

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
            labels.append(label)
            scores_raw.append(float(s_raw))
            scores_hf.append(float(s_hf))
            scores_var.append(float(score_item["pixel_var"]))
            actual_ratios.append(float(score_item["actual_ratio"]))
            core_pixels.append(float(score_item["core_pixels"]))

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "S_raw_pixel": np.asarray(scores_raw, dtype=np.float64),
        "S_high_freq": np.asarray(scores_hf, dtype=np.float64),
        "S_pixel_var": np.asarray(scores_var, dtype=np.float64),
        "actual_ratios": np.asarray(actual_ratios, dtype=np.float64),
        "core_pixels": np.asarray(core_pixels, dtype=np.float64),
    }


def summarize_metrics(calib_data, test_data):
    labels_calib = calib_data["labels"]
    labels_test = test_data["labels"]
    metrics = {}
    for score_name in ["S_raw_pixel", "S_high_freq", "S_pixel_var"]:
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
    set_seed(config.SEED)
    configure_backend()

    exp_dir = os.path.join(config.EXP_DIR, args.exp_name)
    eval_dir = os.path.join(exp_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    log_file = os.path.join(eval_dir, f"genimage_zero_shot_{args.test_split}_K{args.K}.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}", log_file)
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}", log_file)

    _, model, ckpt_path, epoch = load_model(exp_dir, device)
    log(f"加载模型: {ckpt_path} (epoch={epoch})", log_file)

    mask_types = args.mask_types if args.mask_types else config.EVAL_MASK_TYPES
    mask_ratios = args.mask_ratios if args.mask_ratios else config.EVAL_MASK_RATIOS
    generator_paths = list_generators(args.genimage_root, selected=args.generators)
    if len(generator_paths) == 0:
        raise RuntimeError(f"在 {args.genimage_root} 中没有找到可用生成器目录")

    ds_val = RealPatchDataset(split="val", patches_per_image=1)
    dl_val = make_loader(ds_val, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=config.VAL_NUM_WORKERS)
    log("开始估计 tau_r / tau_h ...", log_file)
    tau_dict = estimate_tau(model, dl_val, device, log_file)

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
        ds_calib = build_generator_dataset(generator_path, args.calib_split, preload=args.preload)
        ds_test = build_generator_dataset(generator_path, args.test_split, preload=args.preload)
        dl_calib = make_loader(ds_calib, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dl_test = make_loader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
            for mask_ratio in mask_ratios:
                condition = f"{mask_type}_{int(mask_ratio * 100)}%"
                log(f"  条件 {condition}", log_file)
                calib_data = collect_condition_scores_for_loader(
                    model,
                    dl_calib,
                    mask_type,
                    mask_ratio,
                    tau_dict,
                    args.K,
                    device,
                    desc=f"{generator_name} calib {condition}",
                )
                test_data = collect_condition_scores_for_loader(
                    model,
                    dl_test,
                    mask_type,
                    mask_ratio,
                    tau_dict,
                    args.K,
                    device,
                    desc=f"{generator_name} test {condition}",
                )
                metrics = summarize_metrics(calib_data, test_data)
                metrics["actual_mask_ratio_calib_mean"] = float(np.mean(calib_data["actual_ratios"])) if len(calib_data["actual_ratios"]) > 0 else 0.0
                metrics["actual_mask_ratio_test_mean"] = float(np.mean(test_data["actual_ratios"])) if len(test_data["actual_ratios"]) > 0 else 0.0
                metrics["masked_core_pixels_calib_mean"] = float(np.mean(calib_data["core_pixels"])) if len(calib_data["core_pixels"]) > 0 else 0.0
                metrics["masked_core_pixels_test_mean"] = float(np.mean(test_data["core_pixels"])) if len(test_data["core_pixels"]) > 0 else 0.0
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
    parser = argparse.ArgumentParser(description="5090 Pixel-MAE 在 Genimage-Tiny 上的 zero-shot 分生成器评测")
    parser.add_argument("--exp_name", type=str, default="Exp3_5090_A0")
    parser.add_argument("--genimage_root", type=str, default=r"D:\Genimage-Tiny")
    parser.add_argument("--calib_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--test_split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--K", type=int, default=config.EVAL_K)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=config.EVAL_NUM_WORKERS)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--mask_types", nargs="*", default=None)
    parser.add_argument("--mask_ratios", nargs="*", type=float, default=None)
    parser.add_argument("--generators", nargs="*", default=None)
    args = parser.parse_args()
    evaluate_genimage(args)
