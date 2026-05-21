#!/usr/bin/env python3
import os
import re
import pickle
from pathlib import Path

# ============== CONFIG ==============
# Change this to point to your actual local dataset folder
ROOT = Path("/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/ETH_dataset_CytoTseg")
CLL_DIR   = ROOT / "CLL"
CTRL_DIR  = ROOT / "Control"

IMG_EXT = {".png"}                 
FREQ_RE = re.compile(r'(?<!\d)(400|1500)(?!\d)', re.IGNORECASE)

LOAD_AS_ARRAYS = True              
# The output pickle files will save right alongside your Notebook
OUT_400  = Path("/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/_Data_bank_400_mbar.pkl")
OUT_1500 = Path("/home/ubuntu/CytoTSeg-code/Preparing_ETH_data_for_submission/_Data_bank_1500_mbar.pkl")
IMREAD_FLAG = 0  # cv2.IMREAD_GRAYSCALE
# ===================================


def collect_for_patient(patient_dir: Path, d400: dict, d1500: dict, load_as_arrays: bool):
    """Fill d400 and d1500 with data for one patient."""
    pid = patient_dir.name
    if load_as_arrays:
        import cv2

    for root, dirs, _ in os.walk(patient_dir):
        if "images" not in dirs:
            continue

        run_dir = Path(root)
        m = FREQ_RE.search(run_dir.name) or FREQ_RE.search(run_dir.parent.name)
        if not m:
            continue
        freq = m.group(1)  # "400" or "1500"

        images_dir = run_dir / "images"
        frames = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
        if not frames:
            continue

        target = d400 if freq == "400" else d1500
        if pid not in target:
            target[pid] = {"images": []}

        for p in frames:
            if load_as_arrays:
                im = cv2.imread(str(p), IMREAD_FLAG)
                if im is not None:
                    target[pid]["images"].append(im)
            else:
                target[pid]["images"].append(str(p))


def main():
    data_400, data_1500 = {}, {}

    # CLL patients
    if CLL_DIR.is_dir():
        for d in sorted(CLL_DIR.iterdir()):
            if d.is_dir():
                collect_for_patient(d, data_400, data_1500, LOAD_AS_ARRAYS)

    # Control patients
    if CTRL_DIR.is_dir():
        for d in sorted(CTRL_DIR.iterdir()):
            if d.is_dir():
                collect_for_patient(d, data_400, data_1500, LOAD_AS_ARRAYS)

    # quick summary
    print(f"400Hz: patients={len(data_400)} | total_images={sum(len(v['images']) for v in data_400.values())}")
    print(f"1500Hz: patients={len(data_1500)} | total_images={sum(len(v['images']) for v in data_1500.values())}")

    # save separately
    OUT_400.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_400, "wb") as f:
        pickle.dump(data_400, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUT_1500, "wb") as f:
        pickle.dump(data_1500, f, protocol=pickle.HIGHEST_PROTOCOL)

    # verification print
    if data_400:
        sample = next(iter(data_400.keys()))
        print(f"\nExample 400 patient: {sample}, images={len(data_400[sample]['images'])}")
    if data_1500:
        sample = next(iter(data_1500.keys()))
        print(f"Example 1500 patient: {sample}, images={len(data_1500[sample]['images'])}")

    print(f"\nSaved:\n  400  -> {OUT_400}\n  1500 -> {OUT_1500}")


if __name__ == "__main__":
    main()
