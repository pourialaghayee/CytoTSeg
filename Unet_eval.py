#!/usr/bin/env python3
"""
U-Net Evaluation Script - Compatible with Comprehensive Search Pipeline
Automatically loads correct model architecture from config_unet_v2.json
Supports both DynamicUNet and SmallUNet
"""

import os
import json
import time
import pickle
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# USER CONFIGURATION (EDIT HERE)
# ============================================================================

WORK_DIR   = ""
TEST_PKL   = ""
OUTPUT_PKL = ""
STATS_TXT  = ""

BATCH_SIZE = 32
PAD_MODE   = "reflect"  # reflect|edge|constant
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL ARCHITECTURES (Must match training)
# ============================================================================

def _gn_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.conv1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bn = self.bottleneck(p2)
        u2 = self.up2(bn)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)
        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)
        return self.out(c1)


class DynamicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16, depth=2):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        curr_in = in_channels
        curr_out = base
        for _ in range(depth):
            self.downs.append(DoubleConv(curr_in, curr_out))
            self.pools.append(nn.MaxPool2d(2))
            curr_in = curr_out
            curr_out *= 2
        self.bottleneck = DoubleConv(curr_in, curr_out)
        curr_in = curr_out
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(depth):
            curr_out = curr_in // 2
            self.ups.append(nn.ConvTranspose2d(curr_in, curr_out, 2, 2))
            self.convs.append(DoubleConv(curr_in, curr_out))
            curr_in = curr_out
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.downs[i](x)
            skips.append(x)
            x = self.pools[i](x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(self.depth):
            x = self.ups[i](x)
            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.convs[i](x)
        return self.out(x)


# ============================================================================
# PREPROCESSING
# ============================================================================

def normalize_image(img: np.ndarray, mode: str) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if mode in ("pre_normalized", "none", None):
        return img
    if img.max() > 1.0:
        img = img / 255.0
    if mode == "minmax":
        vmin, vmax = float(img.min()), float(img.max())
        return (img - vmin) / (vmax - vmin) if vmax > vmin else img * 0.0
    if mode == "percentile":
        plow, phigh = np.percentile(img, 1), np.percentile(img, 99)
        if phigh > plow:
            img = np.clip(img, plow, phigh)
            return (img - plow) / (phigh - plow)
        return img * 0.0
    if mode == "zscore":
        mean, std = float(img.mean()), float(img.std())
        return (img - mean) / std if std > 0 else img - mean
    raise ValueError(f"Unknown norm mode: {mode}")


def pad_to_multiple(img: np.ndarray, multiple: int = 4, value: float = 0.0) -> np.ndarray:
    H, W = img.shape
    H_new = int(math.ceil(H / multiple) * multiple)
    W_new = int(math.ceil(W / multiple) * multiple)
    pad_h, pad_w = H_new - H, W_new - W
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=value)


def pad_images_to_common_size(images: List[np.ndarray], pad_mode: str = "reflect") -> List[np.ndarray]:
    Hmax = max(im.shape[0] for im in images)
    Wmax = max(im.shape[1] for im in images)
    out = []
    for im in images:
        H, W = im.shape
        pad_h, pad_w = Hmax - H, Wmax - W
        if pad_h == 0 and pad_w == 0:
            out.append(im)
            continue
        if pad_mode == "constant":
            out.append(np.pad(im, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0))
        elif pad_mode == "reflect":
            try:
                out.append(np.pad(im, ((0, pad_h), (0, pad_w)), mode="reflect"))
            except ValueError:
                out.append(np.pad(im, ((0, pad_h), (0, pad_w)), mode="edge"))
        elif pad_mode == "edge":
            out.append(np.pad(im, ((0, pad_h), (0, pad_w)), mode="edge"))
        else:
            raise ValueError(f"Unknown PAD_MODE: {pad_mode}")
    return out


# ============================================================================
# PREDICTION
# ============================================================================

@torch.no_grad()
def predict_batch(
    images: List[np.ndarray],
    model,
    threshold: float,
    norm_mode: str,
    pad_mode: str
) -> List[np.ndarray]:
    normed = []
    orig_sizes = []
    for im in images:
        im = normalize_image(im, norm_mode)
        orig_sizes.append(im.shape)
        pad_val = float(np.median(im))
        im = pad_to_multiple(im, multiple=4, value=pad_val)
        normed.append(im)
    normed = pad_images_to_common_size(normed, pad_mode=pad_mode)
    batch = np.stack(normed, axis=0).astype(np.float32)
    x = torch.from_numpy(batch)[:, None, ...].to(DEVICE)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()[:, 0, ...]
    preds = []
    for p, (H, W) in zip(probs, orig_sizes):
        p = p[:H, :W]
        m = (p >= threshold).astype(np.uint8) * 255
        preds.append(m)
    return preds


# ============================================================================
# METRICS
# ============================================================================

def dice_iou_precision_recall(pred_u8: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    pred = (pred_u8 > 0).astype(np.float32)
    gt   = (np.asarray(gt) > 0).astype(np.float32)
    tp = float((pred * gt).sum())
    fp = float((pred * (1.0 - gt)).sum())
    fn = float(((1.0 - pred) * gt).sum())
    denom_dice = pred.sum() + gt.sum()
    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else (1.0 if tp == 0 else 0.0)
    denom_iou = pred.sum() + gt.sum() - tp
    iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if tp == 0 else 0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(dice), float(iou), float(precision), float(recall)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_config_and_model(work_dir: str):
    cfg_path = os.path.join(work_dir, "config_unet_v2.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(work_dir, "configunetv2.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found in {work_dir}")

    print(f"Loading config: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model_cfg    = cfg.get("model", {})
    architecture = model_cfg.get("architecture", "SmallUNet")
    base         = int(model_cfg.get("base", 16))
    in_ch        = int(model_cfg.get("in_channels", 1))
    out_ch       = int(model_cfg.get("out_channels", 1))
    depth        = int(model_cfg.get("depth", 2))

    print(f"Architecture: {architecture}")
    print(f"  Base: {base}, Depth: {depth}, Params: {model_cfg.get('params', 'N/A')}")

    if architecture == "DynamicUNet":
        model = DynamicUNet(in_channels=in_ch, out_channels=out_ch, base=base, depth=depth).to(DEVICE)
    elif architecture == "SmallUNet":
        model = SmallUNet(in_channels=in_ch, out_channels=out_ch, base=base).to(DEVICE)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    ckpt = cfg.get("checkpoint_full_path") or cfg.get("checkpointfullpath")
    ckpt_name = cfg.get("checkpoint", "best_model.pth")
    portable_ckpt = os.path.join(work_dir, "checkpoints", ckpt_name)
    if ckpt is None or not os.path.exists(ckpt):
        ckpt = portable_ckpt

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    threshold = float(cfg.get("threshold", 0.5))
    prep      = cfg.get("preprocessing", {})
    norm_mode = prep.get("norm_mode", prep.get("normmode", "pre_normalized"))

    print(f"Threshold: {threshold:.3f}")
    print(f"Normalization mode: {norm_mode}")

    return model, cfg, threshold, norm_mode, ckpt, cfg_path


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_patient_images_and_gts(patient_data: Dict[str, Any]):
    images = None
    for k in ("image", "images", "Image"):
        if k in patient_data:
            images = patient_data[k]
            break
    if images is None:
        raise KeyError(f"Could not find images key. Available: {list(patient_data.keys())}")

    gts = None
    for k in ("Ground_truth", "Groundtruths", "GuckThres", "masks", "mask", "Groundtruth"):
        if k in patient_data:
            gts = patient_data[k]
            break

    return images, gts


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a trained CytoTSeg U-Net on a patient-grouped test pickle."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--test-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--stats-txt", type=Path, help="Runtime TSV output path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pad-mode", choices=("reflect", "edge", "constant"), default="reflect")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None):
    global WORK_DIR, TEST_PKL, OUTPUT_PKL, STATS_TXT, BATCH_SIZE, PAD_MODE, DEVICE
    args = build_parser().parse_args(argv)
    WORK_DIR = str(args.work_dir.expanduser().resolve())
    TEST_PKL = str(args.test_pkl.expanduser().resolve())
    OUTPUT_PKL = str(args.output_pkl.expanduser().resolve())
    STATS_TXT = str(
        args.stats_txt.expanduser().resolve()
        if args.stats_txt
        else Path(OUTPUT_PKL).with_name("inference_stats.tsv")
    )
    BATCH_SIZE = args.batch_size
    PAD_MODE = args.pad_mode
    if BATCH_SIZE < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not Path(TEST_PKL).is_file():
        raise FileNotFoundError(f"Test pickle not found: {TEST_PKL}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(OUTPUT_PKL).parent.mkdir(parents=True, exist_ok=True)
    Path(STATS_TXT).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("U-NET EVALUATION - COMPREHENSIVE SEARCH COMPATIBLE")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Working directory: {WORK_DIR}")
    print(f"Test data: {TEST_PKL}")
    print(f"Output: {OUTPUT_PKL}")
    print(f"Batch size: {BATCH_SIZE}, Pad mode: {PAD_MODE}")
    print()

    model, cfg, threshold, norm_mode, ckpt, cfg_path = load_config_and_model(WORK_DIR)
    print()

    print("Loading test data...")
    with open(TEST_PKL, "rb") as f:
        testdata = pickle.load(f)

    if not isinstance(testdata, dict):
        raise TypeError("Expected testdata to be a dict of patient_id -> patient_data")

    print(f"Found {len(testdata)} patients")
    print()

    per_patient = {}
    all_dice, all_iou = [], []
    patient_stats = []   # <-- new: collect (pid, n_images, runtime_s)

    t0 = time.time()
    for i, (pid, pdata) in enumerate(testdata.items(), 1):
        print(f"[{i}/{len(testdata)}] Processing patient: {pid}")

        images, gts = extract_patient_images_and_gts(pdata)
        n = len(images)
        print(f"  Images: {n}")

        preds = []
        p0 = time.time()

        for j in range(0, n, BATCH_SIZE):
            batch_imgs  = images[j : j + BATCH_SIZE]
            batch_preds = predict_batch(batch_imgs, model, threshold, norm_mode, PAD_MODE)
            preds.extend(batch_preds)

        dt = time.time() - p0
        patient_stats.append((pid, n, dt))   # <-- new

        pdata["Unet_preds"] = [p.astype(np.uint8) for p in preds]

        if gts is not None:
            dices, ious, precs, recs = [], [], [], []
            for pred, gt in zip(preds, gts):
                d, iou, pr, rc = dice_iou_precision_recall(pred, gt)
                dices.append(d)
                ious.append(iou)
                precs.append(pr)
                recs.append(rc)

            per_patient[pid] = {
                "n_images":       n,
                "dice_mean":      float(np.mean(dices)),
                "dice_std":       float(np.std(dices)),
                "iou_mean":       float(np.mean(ious)),
                "iou_std":        float(np.std(ious)),
                "precision_mean": float(np.mean(precs)),
                "recall_mean":    float(np.mean(recs)),
                "time_seconds":   float(dt),
                "ms_per_image":   float(1000.0 * dt / max(1, n)),
            }

            all_dice.extend(dices)
            all_iou.extend(ious)

            print(f"  Dice: {per_patient[pid]['dice_mean']:.4f} ± {per_patient[pid]['dice_std']:.4f}")
            print(f"  IoU:  {per_patient[pid]['iou_mean']:.4f} ± {per_patient[pid]['iou_std']:.4f}")
            print(f"  Time: {dt:.2f}s ({per_patient[pid]['ms_per_image']:.1f} ms/img)")
        else:
            per_patient[pid] = {
                "n_images":     n,
                "time_seconds": float(dt),
                "ms_per_image": float(1000.0 * dt / max(1, n)),
                "note":         "No ground-truth found; only predictions saved.",
            }
            print(f"  Time: {dt:.2f}s ({per_patient[pid]['ms_per_image']:.1f} ms/img)")
            print("  Note: No ground truth available")

        print()

    elapsed    = time.time() - t0
    total_imgs = sum(v["n_images"] for v in per_patient.values())

    # ── write inference stats txt ────────────────────────────────────────────
    print(f"Saving inference stats to: {STATS_TXT}")
    with open(STATS_TXT, "w") as f:
        f.write("patient_id\tn_images\truntime_s\n")
        for pid, n, rt in patient_stats:
            f.write(f"{pid}\t{n}\t{rt:.4f}\n")
        f.write(f"\n# Total images : {total_imgs}\n")
        f.write(f"# Total runtime: {elapsed:.2f}s\n")
        f.write(f"# ms per image : {1000*elapsed/max(1,total_imgs):.2f}\n")
    print("Stats saved!")
    # ────────────────────────────────────────────────────────────────────────

    overall = {
        "n_patients":        len(testdata),
        "n_images_total":    int(total_imgs),
        "threshold":         float(threshold),
        "norm_mode":         norm_mode,
        "batch_size":        int(BATCH_SIZE),
        "pad_mode":          PAD_MODE,
        "time_seconds":      float(elapsed),
        "images_per_second": float(total_imgs / elapsed if elapsed > 0 else 0.0),
        "model_architecture": cfg["model"]["architecture"],
        "model_depth":        cfg["model"]["depth"],
        "model_base":         cfg["model"]["base"],
        "model_params":       cfg["model"]["params"],
    }

    if len(all_dice) > 0:
        overall.update(
            dice_overall_mean=float(np.mean(all_dice)),
            dice_overall_std=float(np.std(all_dice)),
            iou_overall_mean=float(np.mean(all_iou)),
            iou_overall_std=float(np.std(all_iou)),
            dice_per_patient_mean=float(
                np.mean([v["dice_mean"] for v in per_patient.values() if "dice_mean" in v])
            ),
        )

    results      = {"overall": overall, "per_patient": per_patient}
    results_path = os.path.join(WORK_DIR, "evaluation_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total patients: {overall['n_patients']}")
    print(f"Total images:   {overall['n_images_total']}")
    print(f"Processing time: {overall['time_seconds']:.2f}s")
    print(f"Speed: {overall['images_per_second']:.1f} images/sec")
    print()

    if "dice_per_patient_mean" in overall:
        print(f"FINAL DICE (per-patient average): {overall['dice_per_patient_mean']:.4f}")
        print(f"FINAL DICE (all images):          {overall['dice_overall_mean']:.4f} ± {overall['dice_overall_std']:.4f}")
        print(f"FINAL IoU  (all images):          {overall['iou_overall_mean']:.4f} ± {overall['iou_overall_std']:.4f}")

    print()
    print(f"Results saved to: {results_path}")

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(testdata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Predictions saved to: {OUTPUT_PKL}")
    print()
    print("DONE!")


if __name__ == "__main__":
    main()
