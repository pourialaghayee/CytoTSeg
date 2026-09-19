#!/usr/bin/env python3
"""Fine-tune a Cellpose Cyto2 teacher from a trusted annotated pickle."""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from skimage.measure import label
from skimage.util import img_as_float32
from cellpose import models
from cellpose import train as cp_train


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Annotated multi-dataset pickle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model", default="cyto2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset keys to use. Omit to use every dataset in the pickle.",
    )
    parser.add_argument("--model-name", help="Output model name; generated automatically when omitted.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU training.")
    return parser


def collect_samples(db, datasets):
    images, masks = [], []
    for dataset in datasets:
        image_dict = db[dataset]["image"]
        gt_dict = db[dataset]["GT"]
        common_keys = sorted(set(image_dict) & set(gt_dict))
        print(f"\n[{dataset}] Paired samples: {len(common_keys)}")

        kept = 0
        for key in common_keys:
            image = img_as_float32(np.squeeze(image_dict[key]))
            mask = label(np.squeeze(gt_dict[key]) > 0).astype(np.int32)
            if mask.max() == 0:
                continue
            images.append(image)
            masks.append(mask)
            kept += 1
        print(f"[{dataset}] After dropping empty masks: {kept}")
    return images, masks


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_path}")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    with open(input_path, "rb") as handle:
        db = pickle.load(handle)

    datasets = sorted(db) if args.datasets is None else args.datasets
    missing = [name for name in datasets if name not in db]
    if missing:
        raise KeyError(f"Datasets not found: {missing}. Available: {sorted(db)}")
    print(f"Using datasets: {datasets}")

    images, masks = collect_samples(db, datasets)
    if len(images) < 2:
        raise ValueError("At least two non-empty annotated samples are required.")

    indices = list(range(len(images)))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, int(round(args.validation_fraction * len(indices))))
    n_val = min(n_val, len(indices) - 1)

    x_train = [images[i][np.newaxis] for i in indices[n_val:]]
    y_train = [masks[i] for i in indices[n_val:]]
    x_val = [images[i][np.newaxis] for i in indices[:n_val]]
    y_val = [masks[i] for i in indices[:n_val]]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or f"{'_'.join(datasets)}_{args.pretrained_model}_finetuned"
    print(f"Train: {len(x_train)} | Validation: {len(x_val)}")
    print(f"Model name: {model_name}")

    pretrained_path = Path(args.pretrained_model).expanduser()
    if pretrained_path.is_file():
        base_model = models.CellposeModel(
            gpu=not args.cpu,
            pretrained_model=str(pretrained_path.resolve()),
        )
    else:
        base_model = models.CellposeModel(
            gpu=not args.cpu,
            model_type=args.pretrained_model,
        )
    model_path, _, _ = cp_train.train_seg(
        base_model.net,
        train_data=x_train,
        train_labels=y_train,
        test_data=x_val,
        test_labels=y_val,
        channels=[0, 0],
        channel_axis=0,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        normalize=True,
        compute_flows=True,
        rescale=True,
        min_train_masks=1,
        save_path=str(output_dir),
        model_name=model_name,
    )
    print(f"Done. Model saved at: {model_path}")


if __name__ == "__main__":
    main()
