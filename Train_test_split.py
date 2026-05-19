#!/usr/bin/env python3
"""
ETH DC_TSeg Transition Script (Cellpose -> U-Net train/test pickles)
===================================================================
- Reads db[pid][IMAGE_KEY] and db[pid][MASK_KEY]
- Converts instance masks -> binary float32 Ground_truth
- Largest patients (by image count) -> TRAIN, rest -> TEST
- SINGLE_PATIENT_TRAIN mode: exactly 1 CLL + 1 Control (largest) for train
- No cleaning (assumed already done by Cleaning.py)
- Saves: Test_data, Train_data, config.json, split lists, run_log
"""

import os, json, time, pickle, random
from pathlib import Path
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = "/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/ICellCNN"
INPUT_PKL = "patient_data_cyto2_finetuned_clean.pkl"

IMAGE_KEY = "normalized_images"   # or 'images', 'normalized_images
MASK_KEY  = "cyto2_finetuned_clean"

RANDOM_SEED = 42
TEST_RATIO  = 0.95        # fraction of patients used as test per class
                          # e.g. 0.95 with 19 patients -> 1 train, 18 test per class
                          # (ignored when SINGLE_PATIENT_TRAIN = True)

# If True : exactly 1 CLL + 1 Control (largest image count) -> TRAIN, rest -> TEST
# If False: use TEST_RATIO to determine how many largest patients go to TRAIN
SINGLE_PATIENT_TRAIN = True

CLL_PREFIX     = "CLL_"
CONTROL_PREFIX = "Control_"
# =============================================================================


def convert_labeled_to_binary(mask_labeled):
    return (mask_labeled > 0).astype(np.float32)


def get_patient_group(pid):
    if pid.startswith(CLL_PREFIX):     return "CLL"
    if pid.startswith(CONTROL_PREFIX): return "Control"
    return "Other"


def get_image_count(db, pid, image_key):
    """Return number of images for a patient, supporting list or ndarray."""
    data = db[pid][image_key]
    if isinstance(data, np.ndarray):
        return data.shape[0]
    return len(data)


def sort_by_image_count(db, pids, image_key):
    """Return pids sorted descending by image count (largest first)."""
    return sorted(pids, key=lambda p: get_image_count(db, p, image_key), reverse=True)


def build_train_test(db, image_key, mask_key, test_ratio, random_seed,
                     single_patient_train=False):

    random.seed(random_seed)

    # --- collect eligible patients ---
    eligible, skipped = [], []
    for pid in sorted(db.keys()):
        pdata = db[pid]
        if image_key not in pdata:
            skipped.append((pid, f"missing '{image_key}'")); continue
        if mask_key not in pdata:
            skipped.append((pid, f"missing '{mask_key}'")); continue
        imgs  = pdata[image_key]
        masks = pdata[mask_key]
        n_imgs  = imgs.shape[0]  if isinstance(imgs,  np.ndarray) else len(imgs)
        n_masks = masks.shape[0] if isinstance(masks, np.ndarray) else len(masks)
        if not (isinstance(imgs,  (list, np.ndarray)) and
                isinstance(masks, (list, np.ndarray))):
            skipped.append((pid, "images/masks not list or ndarray")); continue
        if n_imgs != n_masks:
            skipped.append((pid, f"len mismatch {n_imgs} vs {n_masks}")); continue
        if get_patient_group(pid) in ("CLL", "Control"):
            eligible.append(pid)
        else:
            skipped.append((pid, "not CLL/Control"))

    cll_pids     = [p for p in eligible if p.startswith(CLL_PREFIX)]
    control_pids = [p for p in eligible if p.startswith(CONTROL_PREFIX)]

    if not cll_pids or not control_pids:
        raise ValueError(f"Need both classes. CLL={len(cll_pids)} Control={len(control_pids)}")

    # --- sort each class by image count DESCENDING (largest first) ---
    cll_pids     = sort_by_image_count(db, cll_pids,     image_key)
    control_pids = sort_by_image_count(db, control_pids, image_key)

    # --- split: largest go to TRAIN, rest to TEST ---
    if single_patient_train:
        # Always pick exactly 1 (the largest) from each class for train
        train_cll     = [cll_pids[0]]
        train_control = [control_pids[0]]
        n_train_per_class = 1
    else:
        # n_train = floor((1 - test_ratio) * n_min), minimum 1
        n_min             = min(len(cll_pids), len(control_pids))
        n_train_per_class = max(1, int(np.floor((1 - test_ratio) * n_min)))
        train_cll     = cll_pids[:n_train_per_class]
        train_control = control_pids[:n_train_per_class]

    train_patients = sorted(train_cll + train_control)
    test_patients  = sorted([p for p in eligible if p not in train_patients])

    test_cll_list     = sorted([p for p in test_patients if p.startswith(CLL_PREFIX)])
    test_control_list = sorted([p for p in test_patients if p.startswith(CONTROL_PREFIX)])

    # --- build test (patient-grouped) ---
    test_data = {}
    for pid in test_patients:
        imgs  = db[pid][image_key]
        masks = db[pid][mask_key]
        imgs_list  = list(imgs)  if isinstance(imgs,  np.ndarray) else imgs
        masks_list = list(masks) if isinstance(masks, np.ndarray) else masks
        test_data[pid] = {
            "image":        imgs_list,
            "Ground_truth": [convert_labeled_to_binary(m) for m in masks_list],
        }

    # --- build train (flat indexed) ---
    train_data = {}
    idx = 0
    for pid in train_patients:
        imgs  = db[pid][image_key]
        masks = db[pid][mask_key]
        imgs_list  = list(imgs)  if isinstance(imgs,  np.ndarray) else imgs
        masks_list = list(masks) if isinstance(masks, np.ndarray) else masks
        for i, (img, m) in enumerate(zip(imgs_list, masks_list)):
            train_data[idx] = {
                "image":        img,
                "Ground_truth": convert_labeled_to_binary(m),
                "pid":          pid,
                "img_index":    int(i),
            }
            idx += 1

    stats = {
        "patients_eligible":     len(eligible),
        "patients_cll":          len(cll_pids),
        "patients_control":      len(control_pids),
        "single_patient_train":  single_patient_train,
        "n_train_per_class":     n_train_per_class,
        "patients_test":         len(test_patients),
        "patients_train":        len(train_patients),
        "images_test":           sum(get_image_count(db, p, image_key) for p in test_patients),
        "images_train":          idx,
        "skipped_patients":      len(skipped),
        # show image counts for selected train patients (sanity check)
        "train_image_counts":    {p: get_image_count(db, p, image_key) for p in train_patients},
    }

    split_info = {
        "train_patients": train_patients,
        "test_patients":  test_patients,
        "test_cll":       test_cll_list,
        "test_control":   test_control_list,
    }

    return test_data, train_data, split_info, stats, skipped


def main():
    base_path  = Path(BASE_PATH)
    input_path = base_path / INPUT_PKL
    assert input_path.exists(), f"Not found: {input_path}"

    out_dir = base_path / "Train_tes_split_smaller_model_normal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    t0 = time.time()
    with open(input_path, "rb") as f:
        db = pickle.load(f)
    load_s = time.time() - t0
    print(f"Loaded {len(db)} patients  ({load_s:.2f}s)")

    # --- build ---
    t1 = time.time()
    test_data, train_data, split_info, stats, skipped = build_train_test(
        db, IMAGE_KEY, MASK_KEY, TEST_RATIO, RANDOM_SEED,
        single_patient_train=SINGLE_PATIENT_TRAIN
    )
    build_s = time.time() - t1

    # --- save pickles ---
    t2 = time.time()
    mode_tag   = "SINGLE" if SINGLE_PATIENT_TRAIN else f"ratio{int(TEST_RATIO * 100)}"
    test_file  = out_dir / f"Test_data_{MASK_KEY}_{mode_tag}.pkl"
    train_file = out_dir / f"Train_data_{MASK_KEY}_{mode_tag}.pkl"

    with open(test_file,  "wb") as f: pickle.dump(test_data,  f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(train_file, "wb") as f: pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_s = time.time() - t2

    # --- save split lists ---
    (out_dir / "train_patients.txt").write_text("\n".join(split_info["train_patients"]) + "\n")
    (out_dir / "test_patients.txt").write_text( "\n".join(split_info["test_patients"])  + "\n")

    # --- save config + stats ---
    config = {
        "input_path":           str(input_path),
        "output_dir":           str(out_dir),
        "image_key":            IMAGE_KEY,
        "mask_key":             MASK_KEY,
        "random_seed":          RANDOM_SEED,
        "test_ratio":           TEST_RATIO,
        "single_patient_train": SINGLE_PATIENT_TRAIN,
    }
    with open(out_dir / "config.json",    "w") as f: json.dump(config, f, indent=2)
    with open(out_dir / "run_stats.json", "w") as f: json.dump({
        "load_s":  load_s,
        "build_s": build_s,
        "save_s":  save_s,
        "stats":   stats,
        "skipped": skipped[:50],
    }, f, indent=2)

    # --- log ---
    if SINGLE_PATIENT_TRAIN:
        mode_str = "SINGLE_PATIENT_TRAIN  (1 largest CLL + 1 largest Control -> train)"
    else:
        mode_str = (f"TEST_RATIO={TEST_RATIO}  "
                    f"(top-{stats['n_train_per_class']} largest per class -> train, rest -> test)")

    train_counts_str = "  ".join(
        f"{p}: {n} imgs" for p, n in stats["train_image_counts"].items()
    )

    log = [
        "TRAIN/TEST SPLIT LOG", "=" * 70,
        f"Input      : {input_path}",
        f"Image key  : {IMAGE_KEY}",
        f"Mask key   : {MASK_KEY}",
        f"Mode       : {mode_str}",
        f"Selection  : largest image count first",
        "",
        f"Eligible patients : {stats['patients_eligible']}",
        f"  CLL              : {stats['patients_cll']}",
        f"  Control          : {stats['patients_control']}",
        "",
        f"Train patients    : {stats['patients_train']}  -> {split_info['train_patients']}",
        f"Train img counts  : {train_counts_str}",
        f"Test  patients    : {stats['patients_test']}",
        "",
        f"Images in train   : {stats['images_train']}",
        f"Images in test    : {stats['images_test']}",
        "",
        f"Skipped patients  : {stats['skipped_patients']}",
        "",
        "Output files:",
        f"  {test_file.name}",
        f"  {train_file.name}",
        "  train_patients.txt",
        "  test_patients.txt",
        "  config.json",
        "  run_stats.json",
    ]
    log_str = "\n".join(log)
    (out_dir / "run_log.txt").write_text(log_str)
    print(log_str)
    print(f"\nTiming: load={load_s:.2f}s  build={build_s:.2f}s  save={save_s:.2f}s")
    print("DONE.")


if __name__ == "__main__":
    main()

