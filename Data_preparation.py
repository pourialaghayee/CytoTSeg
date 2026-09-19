#!/usr/bin/env python3
import os
import re
import pickle
import argparse
from pathlib import Path

IMG_EXT = {".png"}                 
FREQ_RE = re.compile(r'(?<!\d)(400|1500)(?!\d)', re.IGNORECASE)
IMREAD_FLAG = 0  # cv2.IMREAD_GRAYSCALE


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
        frames = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT
        )
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect ETH-CLL PNG images into 400 Hz and 1500 Hz pickle files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset directory containing CLL/ and Control/ patient folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the two prepared pickle files are written.",
    )
    parser.add_argument("--output-400", default="data_400.pkl")
    parser.add_argument("--output-1500", default="data_1500.pkl")
    parser.add_argument(
        "--store-paths",
        action="store_true",
        help="Store image paths instead of loading image arrays (not suitable for later pipeline stages).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cll_dir = root / "CLL"
    ctrl_dir = root / "Control"
    load_as_arrays = not args.store_paths

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not cll_dir.is_dir() and not ctrl_dir.is_dir():
        raise FileNotFoundError(
            f"Expected at least one of {cll_dir} or {ctrl_dir}. "
            "See the README dataset layout."
        )

    data_400, data_1500 = {}, {}

    # CLL patients
    if cll_dir.is_dir():
        for d in sorted(cll_dir.iterdir()):
            if d.is_dir():
                collect_for_patient(d, data_400, data_1500, load_as_arrays)

    # Control patients
    if ctrl_dir.is_dir():
        for d in sorted(ctrl_dir.iterdir()):
            if d.is_dir():
                collect_for_patient(d, data_400, data_1500, load_as_arrays)

    # quick summary
    print(f"400Hz: patients={len(data_400)} | total_images={sum(len(v['images']) for v in data_400.values())}")
    print(f"1500Hz: patients={len(data_1500)} | total_images={sum(len(v['images']) for v in data_1500.values())}")

    # save separately
    output_dir.mkdir(parents=True, exist_ok=True)
    out_400 = output_dir / args.output_400
    out_1500 = output_dir / args.output_1500
    with open(out_400, "wb") as f:
        pickle.dump(data_400, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_1500, "wb") as f:
        pickle.dump(data_1500, f, protocol=pickle.HIGHEST_PROTOCOL)

    # verification print
    if data_400:
        sample = next(iter(data_400.keys()))
        print(f"\nExample 400 patient: {sample}, images={len(data_400[sample]['images'])}")
    if data_1500:
        sample = next(iter(data_1500.keys()))
        print(f"Example 1500 patient: {sample}, images={len(data_1500[sample]['images'])}")

    print(f"\nSaved:\n  400  -> {out_400}\n  1500 -> {out_1500}")


if __name__ == "__main__":
    main()
