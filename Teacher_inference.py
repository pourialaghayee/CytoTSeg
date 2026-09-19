#!/usr/bin/env python3
"""Generate Cellpose instance-mask pseudo-labels for a CytoTSeg patient pickle."""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from cellpose import models


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Normalized patient pickle.")
    parser.add_argument("--output", type=Path, required=True, help="Pickle with teacher masks added.")
    parser.add_argument("--image-key", default="normalized_images")
    parser.add_argument("--output-key", default="cyto2_finetuned")
    parser.add_argument("--model", default="cyto2", help="Cellpose model name or trained model path.")
    parser.add_argument("--diameter", type=float, help="Expected object diameter; omit for auto estimation.")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU inference.")
    parser.add_argument(
        "--no-cellpose-normalize",
        action="store_true",
        help="Disable Cellpose's internal normalization (input is already percentile-normalized).",
    )
    return parser


def create_model(model_name_or_path, use_gpu):
    model_path = Path(model_name_or_path).expanduser()
    if model_path.is_file():
        return models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path.resolve()))

    # Cellpose 3 uses model_type for built-in models. Some compatible releases
    # accept the same name through pretrained_model, so retain a fallback.
    try:
        return models.CellposeModel(gpu=use_gpu, model_type=model_name_or_path)
    except TypeError:
        return models.CellposeModel(gpu=use_gpu, pretrained_model=model_name_or_path)


def run_eval(model, images, args):
    kwargs = {
        "diameter": args.diameter,
        "channels": [0, 0],
        "normalize": not args.no_cellpose_normalize,
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
    }
    result = model.eval(images, **kwargs)
    masks = result[0] if isinstance(result, tuple) else result
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        masks = [masks]
    return [np.asarray(mask, dtype=np.int32) for mask in masks]


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_path}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")

    with open(input_path, "rb") as handle:
        db = pickle.load(handle)
    if not isinstance(db, dict):
        raise TypeError("Expected a dictionary of patient records.")

    print(f"Loading Cellpose teacher: {args.model}")
    model = create_model(args.model, use_gpu=not args.cpu)
    started = time.perf_counter()
    total = 0

    for patient_index, (patient_id, patient_data) in enumerate(db.items(), 1):
        if args.image_key not in patient_data:
            raise KeyError(
                f"Patient '{patient_id}' has no '{args.image_key}' key. "
                f"Available keys: {sorted(patient_data)}"
            )
        images = list(patient_data[args.image_key])
        predictions = []
        patient_started = time.perf_counter()
        for offset in range(0, len(images), args.chunk_size):
            chunk = images[offset:offset + args.chunk_size]
            predictions.extend(run_eval(model, chunk, args))

        if len(predictions) != len(images):
            raise RuntimeError(
                f"Teacher returned {len(predictions)} masks for {len(images)} images "
                f"in patient '{patient_id}'."
            )
        patient_data[args.output_key] = predictions
        total += len(images)
        elapsed = time.perf_counter() - patient_started
        print(
            f"[{patient_index}/{len(db)}] {patient_id}: {len(images)} images, "
            f"{elapsed:.2f}s"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    print(f"Saved {total} pseudo-labels to: {output_path}")
    print(f"Total inference time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
