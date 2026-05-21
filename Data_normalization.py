#!/usr/bin/env python3
"""
Image Pre-normalization Script
- Global percentile normalization (no tiling — avoids boundary artifacts)
- Mimics Cellpose internal normalization
- Saves db[pid]['normalized_images'] as float32 [0, 1]
"""

import os
import time
import pickle
import numpy as np
import cv2
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH      = "/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/"
INPUT_PKL       = "_Data_bank_400_mbar.pkl"
OUTPUT_PKL      = "_Data_bank_400_mbar_normalized.pkl"

APPLY_CLAHE     = False    # No CLAHE — global percentile is enough for flow cytometry
PERCENTILE_LOW  = 1.0      # Cellpose default
PERCENTILE_HIGH = 99.0     # Cellpose default
TILE_SIZE       = 0        # 0 = global normalization (no tiling, no boundary artifacts)
OUTPUT_DTYPE    = 'float32'
NUM_WORKERS     = 16
PROGRESS_EVERY  = 5


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def percentile_normalize(img: np.ndarray,
                         percentile_low: float = 1.0,
                         percentile_high: float = 99.0,
                         tile_size: int = 0) -> np.ndarray:
    """
    Normalize image using global percentile clipping.
    Mimics Cellpose internal normalization exactly.

    tile_size=0  → global (recommended for flow/DC cytometry)
    tile_size>0  → per-tile (only useful for uneven widefield illumination)
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if tile_size > 0 and (img.shape[0] > tile_size or img.shape[1] > tile_size):
        # Tile-based (kept for compatibility but NOT recommended for this data)
        normalized = np.zeros_like(img, dtype=np.float32)
        H, W = img.shape
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                tile = img[y:y_end, x:x_end]
                p_low  = np.percentile(tile, percentile_low)
                p_high = np.percentile(tile, percentile_high)
                if p_high > p_low:
                    tile_norm = np.clip((tile - p_low) / (p_high - p_low), 0, 1)
                else:
                    tile_norm = np.zeros_like(tile, dtype=np.float32)
                normalized[y:y_end, x:x_end] = tile_norm
        return normalized

    else:
        # Global normalization — no tile boundaries, consistent across image sizes
        p_low  = np.percentile(img, percentile_low)
        p_high = np.percentile(img, percentile_high)
        if p_high > p_low:
            return np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
        else:
            return np.zeros_like(img, dtype=np.float32)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on 8-bit grayscale (optional — disabled by default)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        m = float(img.max())
        img = (img.astype(np.float32) / max(m, 1e-6) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def normalize_single_image(img: np.ndarray,
                            apply_clahe: bool = False,
                            percentile_low: float = 1.0,
                            percentile_high: float = 99.0,
                            tile_size: int = 0,
                            output_dtype: str = 'float32') -> np.ndarray:
    """Normalize a single image. Grayscale conversion → optional CLAHE → percentile norm."""
    # Grayscale
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Optional CLAHE
    if apply_clahe:
        img_gray = enhance_contrast(img_gray)

    # Normalize
    normalized = percentile_normalize(img_gray, percentile_low, percentile_high, tile_size)

    if output_dtype == 'uint8':
        return (normalized * 255).astype(np.uint8)
    return normalized.astype(np.float32)


# ============================================================================
# DATABASE NORMALIZATION
# ============================================================================

def prenormalize_database(db: Dict[str, Any],
                          *,
                          apply_clahe: bool = False,
                          percentile_low: float = 1.0,
                          percentile_high: float = 99.0,
                          tile_size: int = 0,
                          output_dtype: str = 'float32',
                          num_workers: int = 8,
                          progress_every: int = 5) -> Dict[str, Any]:
    """
    Pre-normalize all images in the database.
    Adds db[pid]['normalized_images'] (float32, [0,1]) and db[pid]['normalization_info'].
    """
    total_start    = time.perf_counter()
    total_images   = 0
    total_patients = len(db)

    print("=" * 70)
    print("PRE-NORMALIZING DATABASE IMAGES")
    print("=" * 70)
    print(f"  CLAHE        : {'ON' if apply_clahe else 'OFF'}")
    print(f"  Percentiles  : [{percentile_low}, {percentile_high}]")
    print(f"  Tiling       : {'global (no tiling)' if tile_size == 0 else f'{tile_size}px tiles'}")
    print(f"  Output dtype : {output_dtype}")
    print(f"  Workers      : {num_workers}")
    print("=" * 70)
    print()

    for idx, (pid, pdata) in enumerate(db.items(), 1):
        t0   = time.perf_counter()
        imgs = pdata.get('images', [])
        n    = len(imgs)

        if n == 0:
            db[pid]['normalized_images']  = []
            db[pid]['normalization_info'] = {
                'apply_clahe': apply_clahe, 'percentile_low': percentile_low,
                'percentile_high': percentile_high, 'tile_size': tile_size,
                'output_dtype': output_dtype, 'num_images': 0, 'seconds_total': 0.0
            }
            continue

        # Parallel normalization
        if num_workers > 1:
            normalized_imgs = [None] * n
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        normalize_single_image,
                        img, apply_clahe, percentile_low, percentile_high, tile_size, output_dtype
                    ): i for i, img in enumerate(imgs)
                }
                for future in as_completed(futures):
                    normalized_imgs[futures[future]] = future.result()
        else:
            normalized_imgs = [
                normalize_single_image(img, apply_clahe, percentile_low,
                                        percentile_high, tile_size, output_dtype)
                for img in imgs
            ]

        elapsed       = time.perf_counter() - t0
        total_images += n

        db[pid]['normalized_images']  = normalized_imgs
        db[pid]['normalization_info'] = {
            'apply_clahe':       apply_clahe,
            'percentile_low':    percentile_low,
            'percentile_high':   percentile_high,
            'tile_size':         tile_size,
            'output_dtype':      output_dtype,
            'num_images':        n,
            'seconds_total':     float(elapsed),
            'seconds_per_image': float(elapsed / n),
        }

        if idx % progress_every == 0 or idx == total_patients:
            print(f"[{idx}/{total_patients}] {pid}: {n} images in "
                  f"{elapsed:.2f}s ({elapsed/n*1000:.1f}ms/img)")

    total_elapsed = time.perf_counter() - total_start
    print()
    print("=" * 70)
    print(f"DONE  — {total_patients} patients, {total_images} images, "
          f"{total_elapsed:.2f}s total ({total_elapsed/max(total_images,1)*1000:.2f}ms/img avg)")
    print("=" * 70)
    return db


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    start = time.perf_counter()

    input_path = os.path.join(INPUT_PATH, INPUT_PKL)
    output_path = os.path.join(INPUT_PATH, OUTPUT_PKL)

    print(f"Loading: {input_path}")
    t0 = time.perf_counter()
    with open(input_path, "rb") as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} patients in {time.perf_counter()-t0:.2f}s\n")

    db = prenormalize_database(
        db,
        apply_clahe     = APPLY_CLAHE,
        percentile_low  = PERCENTILE_LOW,
        percentile_high = PERCENTILE_HIGH,
        tile_size       = TILE_SIZE,
        output_dtype    = OUTPUT_DTYPE,
        num_workers     = NUM_WORKERS,
        progress_every  = PROGRESS_EVERY,
    )

    print(f"\nSaving: {output_path}")
    t1 = time.perf_counter()
    with open(output_path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved in {time.perf_counter()-t1:.2f}s")

    print(f"\nTotal elapsed: {time.perf_counter()-start:.2f}s")
    print(f"Output: {output_path}")
