#!/usr/bin/env python3
"""
Cleans masks in db[pid]['Unet_preds'] and writes a separate output by default.

For guck2022 / guck2025: additionally removes objects whose bounding box
centroid or extent overlaps the dark border zones (rows < TOP_DEAD_ROWS
or rows > BOTTOM_DEAD_ROW).
"""

import time, pickle, math, argparse
from pathlib import Path
import numpy as np
import cv2


# ============================================================
# UNIFIED DATASET CONFIG
# ============================================================
DATASET_CONFIG = {
    "eth1500":  dict(diameter=13.7,  min_area_px=75,   min_circ=0.60, max_ar=4.0,  max_def=0.40, edge_filter=True),
    "eth400":   dict(diameter=13.5,  min_area_px=75,   min_circ=0.60, max_ar=4.0,  max_def=0.40, edge_filter=True),
    "guck2022": dict(diameter=26.6,  min_area_px=160,  min_circ=0.80, max_ar=5.0,  max_def=0.45, edge_filter=False),
    "guck2025": dict(diameter=22.2,  min_area_px=150,  min_circ=0.25, max_ar=12.0, max_def=0.6,  edge_filter=False),
    "icellcnn": dict(diameter=73.0,  min_area_px=2500, min_circ=0.85, max_ar=4.0,  max_def=0.40, edge_filter=False),
}


# ============================================================
# DEAD-ZONE CONFIG  (top/bottom border rows to exclude)
# Applied only to datasets listed here.
# Objects whose bounding box overlaps the dead zone are dropped.
# ============================================================
DEAD_ZONE_CONFIG = {
    # "dataset": (top_dead_rows, bottom_dead_row)
    # top_dead_rows  : rows  0 .. top_dead_rows-1  are dead  (objects crossing into this zone are dropped)
    # bottom_dead_row: rows  bottom_dead_row .. H-1 are dead  (objects crossing into this zone are dropped)
    "guck2025": (14, 65),
    "guck2022": (14, 65),
}


EDGE_BUFFER     = 2
MAX_AREA_FRAC   = 0.25
DROP_WIDE_BANDS = True
MIN_WIDTH_FRAC  = 0.70
MIN_BAND_ASPECT = 6.0

PROGRESS_EVERY = 100


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


def _in_dead_zone(x, y, w, h, top_dead, bottom_dead):
    """
    Returns True if the bounding box overlaps the dead zone at all.
    Dead zone = rows [0, top_dead) or rows [bottom_dead, H).

    Any object whose bbox top edge (y) is above top_dead  →  drop.
    Any object whose bbox bottom edge (y+h) reaches bottom_dead or beyond  →  drop.
    """
    obj_top    = y
    obj_bottom = y + h          # exclusive lower edge of bbox

    touches_top    = obj_top    < top_dead
    touches_bottom = obj_bottom > bottom_dead

    return touches_top or touches_bottom


def _keep_object(bm_u8, params, H, W, dead_zone=None):
    area_px = int(bm_u8.sum())
    if area_px < params['min_area_px']:
        return False

    ctrs, _ = cv2.findContours(bm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ctrs:
        return False
    contour  = max(ctrs, key=cv2.contourArea)
    raw_area = cv2.contourArea(contour)
    perim    = cv2.arcLength(contour, True)

    circ   = (4 * math.pi * raw_area / perim ** 2) if perim > 0 else 0.0
    deform = (max(0.0, 1 - 2 * math.sqrt(math.pi * raw_area) / perim)
              if perim > 0 and raw_area > 0 else 0.0)
    ar     = _aspect_ratio(contour)

    x, y, w, h = cv2.boundingRect(contour)

    # ── Dead-zone filter (guck2022 / guck2025) ──────────────
    if dead_zone is not None:
        top_dead, bottom_dead = dead_zone
        if _in_dead_zone(x, y, w, h, top_dead, bottom_dead):
            return False

    # ── Edge / wide-band filter (eth datasets) ──────────────
    if params.get('edge_filter', False):
        touches = _touches_edges(x, y, w, h, H, W)
        if area_px >= MAX_AREA_FRAC * (H * W) and touches >= 2:
            return False
        if DROP_WIDE_BANDS and touches >= 1:
            if w >= MIN_WIDTH_FRAC * W and (w / max(h, 1)) >= MIN_BAND_ASPECT:
                return False

    if circ  < params['min_circ']:                 return False
    if np.isfinite(ar) and ar > params['max_ar']:  return False
    if deform > params['max_def']:                 return False

    return True


def clean_instance_map(inst_map, params, dead_zone=None):
    inst      = inst_map.astype(np.int32)
    foreground_labels = np.unique(inst[inst > 0])
    if len(foreground_labels) == 1:
        # U-Net output is binary (usually 0/255); label disconnected objects first.
        _, inst = cv2.connectedComponents((inst > 0).astype(np.uint8))
        inst = inst.astype(np.int32)
    H, W      = inst.shape
    out       = np.zeros((H, W), dtype=np.int32)
    new_label = 1

    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        bm = (inst == cell_id).astype(np.uint8)
        if _keep_object(bm, params, H, W, dead_zone=dead_zone):
            out[bm == 1] = new_label
            new_label += 1

    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description="Clean student prediction masks without overwriting the source by default."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input prediction pickle.")
    parser.add_argument("--output", type=Path, help="Output pickle. Defaults to <input>_cleaned.pkl.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input pickle.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--input-key", default="Unet_preds")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    t_start = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_path}")
    if args.in_place and args.output is not None:
        raise ValueError("Use either --in-place or --output, not both.")
    output_path = (
        input_path if args.in_place
        else (args.output.expanduser().resolve() if args.output
              else input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}"))
    )

    params    = DATASET_CONFIG[args.dataset]
    dead_zone = DEAD_ZONE_CONFIG.get(args.dataset, None)   # None for datasets without dead zones

    print("=" * 70)
    print(f"INSTANCE MAP CLEANER — key '{args.input_key}'")
    print(f"Dataset    : {args.dataset}")
    print(f"min_area   : {params['min_area_px']}")
    print(f"min_circ   : {params['min_circ']}")
    print(f"max_ar     : {params['max_ar']}")
    print(f"max_def    : {params['max_def']}")
    print(f"edge_filter: {params['edge_filter']}")
    if dead_zone:
        print(f"dead_zone  : rows 0–{dead_zone[0]-1} (top)  |  rows {dead_zone[1]}–H (bottom)")
    else:
        print(f"dead_zone  : disabled")
    print("=" * 70)

    with open(input_path, 'rb') as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} patients\n")

    for pid, pdata in db.items():
        masks_in = pdata.get(args.input_key, [])
        n        = len(masks_in)

        if n == 0:
            print(f"  [{pid}] no masks — skipping")
            continue

        print(f"  [{pid}] n={n}")
        masks_out           = []
        t0                  = time.perf_counter()
        total_in, total_out = 0, 0

        for i, mask in enumerate(masks_in, 1):
            mask_np = np.asarray(mask)
            n_in    = int((np.unique(mask_np) > 0).sum())
            cleaned = clean_instance_map(mask_np, params, dead_zone=dead_zone)
            n_out   = int((np.unique(cleaned) > 0).sum())
            masks_out.append(cleaned)
            total_in  += n_in
            total_out += n_out

            if i % args.progress_every == 0 or i == n:
                print(f"    [{i}/{n}]  {time.perf_counter()-t0:.1f}s elapsed")

        db[pid][args.input_key] = masks_out
        pct = 100 * total_out / max(total_in, 1)
        print(f"  Done in {time.perf_counter()-t0:.2f}s  "
              f"cells: {total_in} → {total_out} ({pct:.1f}% kept)\n")

    print(f"Saving: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {output_path}")
    print(f"Total time: {time.perf_counter()-t_start:.2f}s")


if __name__ == "__main__":
    main()
