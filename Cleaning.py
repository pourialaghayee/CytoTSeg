#!/usr/bin/env python3
"""
ICellCNN - Instance Map Cleaner
Reads  db[pid][INPUT_KEY]  (list of int32 instance masks)
Cleans each object separately — keeps good, removes bad
Saves  db[pid][OUTPUT_KEY] (list of int32 instance masks, relabeled 1..M)
Output stays as instance map — NOT converted to binary
"""

import time, pickle, math
import numpy as np
import cv2

# ============================================================
# UNIFIED DATASET CONFIG — all thresholds + flags in one place
# ============================================================
DATASET_CONFIG = {
    "eth1500":  dict(diameter=13.7,  min_area_px=75,  min_circ=0.60, max_ar=4.0,  max_def=0.40, edge_filter=True),
    "eth400":   dict(diameter=13.5,  min_area_px=75,  min_circ=0.60, max_ar=4.0,  max_def=0.40, edge_filter=True),
    "guck2022": dict(diameter=26.6,  min_area_px=160,  min_circ=0.80, max_ar=5.0,  max_def=0.45, edge_filter=False),
    "guck2025": dict(diameter=22.2,  min_area_px=150,  min_circ=0.25, max_ar=12.0, max_def=0.6, edge_filter=False),
    "icellcnn": dict(diameter=73.0,  min_area_px=2500, min_circ=0.85, max_ar=4.0,  max_def=0.40, edge_filter=False),
}
# ============== CONFIG ==============
DATA_PKL   = "/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/_Data_bank_400_mbar_normalized.pkl"
OUT_PKL    = "/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/_Data_bank_400_mbar_normalized_cleaned.pkl"
INPUT_KEY  = "cyto2_finetuned"
OUTPUT_KEY = "cyto2_finetuned_clean"

DATASET    = "eth400"   # ← change this to switch dataset

# Border filters (only active when edge_filter=True in DATASET_CONFIG)
EDGE_BUFFER     = 2
MAX_AREA_FRAC   = 0.25   # drop if area >= 25% of image AND touches >= 2 borders
DROP_WIDE_BANDS = True
MIN_WIDTH_FRAC  = 0.70   # drop if width >= 70% of image width...
MIN_BAND_ASPECT = 6.0    # ...and aspect ratio >= 6

PROGRESS_EVERY = 100
# ====================================


def _touches_edges(x, y, w, h, H, W):
    return sum([
        y <= EDGE_BUFFER,
        y + h >= H - EDGE_BUFFER,
        x <= EDGE_BUFFER,
        x + w >= W - EDGE_BUFFER,
    ])


def _aspect_ratio(contour):
    if contour is None or len(contour) == 0:
        return np.nan
    if len(contour) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(contour)
        major = max(MA, ma)
        minor = max(1e-6, min(MA, ma))
        return major / minor
    _, (w, h), _ = cv2.minAreaRect(contour)
    return max(w, h) / max(1e-6, min(w, h))


def _keep_object(bm_u8, params, H, W):
    """
    bm_u8 : {0,1} uint8 mask of ONE object
    params : dict from DATASET_CONFIG
    Returns True if object passes all filters
    """
    # --- Area ---
    area_px = int(bm_u8.sum())
    if area_px < params['min_area_px']:
        return False

    # --- Contour ---
    ctrs, _ = cv2.findContours(bm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ctrs:
        return False
    contour  = max(ctrs, key=cv2.contourArea)
    raw_area = cv2.contourArea(contour)
    perim    = cv2.arcLength(contour, True)

    # --- Shape features ---
    circ   = (4 * math.pi * raw_area / perim ** 2) if perim > 0 else 0.0
    deform = (max(0.0, 1 - 2 * math.sqrt(math.pi * raw_area) / perim)
              if perim > 0 and raw_area > 0 else 0.0)
    ar     = _aspect_ratio(contour)

    # --- Border filters (only if edge_filter=True for this dataset) ---
    if params.get('edge_filter', False):
        x, y, w, h = cv2.boundingRect(contour)
        touches    = _touches_edges(x, y, w, h, H, W)

        # Large border blob
        if area_px >= MAX_AREA_FRAC * (H * W) and touches >= 2:
            return False

        # Wide edge band
        if DROP_WIDE_BANDS and touches >= 1:
            if w >= MIN_WIDTH_FRAC * W and (w / max(h, 1)) >= MIN_BAND_ASPECT:
                return False

    # --- Shape filters ---
    if circ  < params['min_circ']:                  return False
    if np.isfinite(ar) and ar > params['max_ar']:   return False
    if deform > params['max_def']:                  return False

    return True


def clean_instance_map(inst_map, params):
    """
    inst_map : int32 label map (0=bg, 1..N=cells)
    Returns  : int32 label map — bad objects removed, survivors relabeled 1..M
    Instance identity preserved via relabeling — NOT converted to binary
    """
    inst      = inst_map.astype(np.int32)
    H, W      = inst.shape
    out       = np.zeros((H, W), dtype=np.int32)
    new_label = 1

    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        bm = (inst == cell_id).astype(np.uint8)
        if _keep_object(bm, params, H, W):
            out[bm == 1] = new_label
            new_label += 1

    return out


def main():
    t_start = time.perf_counter()

    assert DATASET in DATASET_CONFIG, f"Unknown dataset '{DATASET}' — add it to DATASET_CONFIG"
    params = DATASET_CONFIG[DATASET]

    print("=" * 70)
    print(f"INSTANCE MAP CLEANER — {INPUT_KEY} → {OUTPUT_KEY}")
    print(f"Dataset    : {DATASET}")
    print(f"min_area   : {params['min_area_px']}")
    print(f"min_circ   : {params['min_circ']}")
    print(f"max_ar     : {params['max_ar']}")
    print(f"max_def    : {params['max_def']}")
    print(f"edge_filter: {params['edge_filter']}")
    print("=" * 70)

    with open(DATA_PKL, 'rb') as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} patients\n")

    for pid, pdata in db.items():
        masks_in = pdata.get(INPUT_KEY, [])
        n        = len(masks_in)

        if n == 0:
            db[pid][OUTPUT_KEY] = []
            print(f"  [{pid}] no masks — skipping")
            continue

        print(f"  [{pid}] n={n}")
        masks_out           = []
        t0                  = time.perf_counter()
        total_in, total_out = 0, 0

        for i, mask in enumerate(masks_in, 1):
            mask_np  = np.asarray(mask)
            n_in     = int((np.unique(mask_np) > 0).sum())
            cleaned  = clean_instance_map(mask_np, params)
            n_out    = int((np.unique(cleaned) > 0).sum())
            masks_out.append(cleaned)
            total_in  += n_in
            total_out += n_out

            if i % PROGRESS_EVERY == 0 or i == n:
                print(f"    [{i}/{n}]  {time.perf_counter()-t0:.1f}s elapsed")

        db[pid][OUTPUT_KEY] = masks_out
        pct = 100 * total_out / max(total_in, 1)
        print(f"  Done in {time.perf_counter()-t0:.2f}s  "
              f"cells: {total_in} → {total_out} ({pct:.1f}% kept)\n")

    print("Saving ...")
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {OUT_PKL}")
    print(f"Total time: {time.perf_counter()-t_start:.2f}s")


if __name__ == "__main__":
    main()
