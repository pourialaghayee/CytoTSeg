#!/usr/bin/env python3
"""Run the CytoTSeg workflow from one JSON configuration file."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    outputs: tuple[Path, ...]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "prepare", "train", "evaluate", "show-config"),
        help="Pipeline section to run.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Pipeline JSON configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Run stages even when their outputs exist.")
    parser.add_argument("--from-step", help="Start at this stage name (only within the chosen command).")
    parser.add_argument("--to-step", help="Stop after this stage name (only within the chosen command).")
    return parser


def load_config(path: Path):
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config, config_path


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def add_flag(command: list[str], flag: str, enabled: bool):
    if enabled:
        command.append(flag)


def require(config: dict, dotted_key: str):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[key]
    return value


def make_stages(config: dict, config_path: Path) -> list[Stage]:
    config_dir = config_path.parent
    raw_root = resolve_path(require(config, "data.raw_data_root"), config_dir)
    work_dir = resolve_path(require(config, "run.work_dir"), config_dir)
    frequency = int(config.get("data", {}).get("frequency", 400))
    if frequency not in (400, 1500):
        raise ValueError("data.frequency must be 400 or 1500.")

    dataset = config.get("data", {}).get("dataset", f"eth{frequency}")
    image_key = config.get("data", {}).get("image_key", "normalized_images")
    teacher_key = config.get("teacher", {}).get("output_key", "cyto2_finetuned")
    clean_key = config.get("teacher", {}).get("clean_key", "cyto2_finetuned_clean")
    device = config.get("run", {}).get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("run.device must be auto, cpu, or cuda.")

    prepared_dir = work_dir / "00_prepared"
    prepared_400 = prepared_dir / "data_400.pkl"
    prepared_1500 = prepared_dir / "data_1500.pkl"
    prepared_selected = prepared_400 if frequency == 400 else prepared_1500
    normalized = work_dir / "01_normalized" / f"data_{frequency}_normalized.pkl"
    teacher_output = work_dir / "02_teacher" / f"data_{frequency}_teacher.pkl"
    cleaned = work_dir / "03_cleaned" / f"data_{frequency}_teacher_clean.pkl"
    split_dir = work_dir / "04_split"
    train_pkl = split_dir / "train.pkl"
    test_pkl = split_dir / "test.pkl"
    train_dir = work_dir / "05_student"
    predictions = work_dir / "06_predictions" / "test_unet_preds.pkl"
    cleaned_predictions = work_dir / "06_predictions" / "test_unet_preds_cleaned.pkl"
    morphology_dir = work_dir / "07_morphology"
    morphology_db = morphology_dir / "test_morphology.pkl"
    classification_dir = work_dir / "08_classification"

    normalization = config.get("normalization", {})
    teacher = config.get("teacher", {})
    split = config.get("split", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    morphology = config.get("morphology", {})
    classification = config.get("classification", {})

    prepare_cmd = [
        sys.executable,
        str(REPO_ROOT / "Data_preparation.py"),
        "--data-root", str(raw_root),
        "--output-dir", str(prepared_dir),
    ]

    normalize_cmd = [
        sys.executable,
        str(REPO_ROOT / "Data_normalization.py"),
        "--input", str(prepared_selected),
        "--output", str(normalized),
        "--percentile-low", str(normalization.get("percentile_low", 1.0)),
        "--percentile-high", str(normalization.get("percentile_high", 99.0)),
        "--tile-size", str(normalization.get("tile_size", 0)),
        "--output-dtype", str(normalization.get("output_dtype", "float32")),
        "--workers", str(normalization.get("workers", 8)),
    ]
    add_flag(normalize_cmd, "--clahe", bool(normalization.get("clahe", False)))

    teacher_cmd = [
        sys.executable,
        str(REPO_ROOT / "Teacher_inference.py"),
        "--input", str(normalized),
        "--output", str(teacher_output),
        "--image-key", image_key,
        "--output-key", teacher_key,
        "--model", str(teacher.get("model", "cyto2")),
        "--chunk-size", str(teacher.get("chunk_size", 256)),
        "--flow-threshold", str(teacher.get("flow_threshold", 0.4)),
        "--cellprob-threshold", str(teacher.get("cellprob_threshold", 0.0)),
    ]
    if teacher.get("diameter") is not None:
        teacher_cmd.extend(["--diameter", str(teacher["diameter"])])
    add_flag(teacher_cmd, "--cpu", device == "cpu")
    add_flag(
        teacher_cmd,
        "--no-cellpose-normalize",
        bool(teacher.get("disable_internal_normalization", False)),
    )

    clean_cmd = [
        sys.executable,
        str(REPO_ROOT / "Cleaning.py"),
        "--input", str(teacher_output),
        "--output", str(cleaned),
        "--dataset", dataset,
        "--input-key", teacher_key,
        "--output-key", clean_key,
    ]

    split_cmd = [
        sys.executable,
        str(REPO_ROOT / "Train_test_split.py"),
        "--input", str(cleaned),
        "--output-dir", str(split_dir),
        "--train-output", train_pkl.name,
        "--test-output", test_pkl.name,
        "--image-key", image_key,
        "--mask-key", clean_key,
        "--random-seed", str(split.get("random_seed", 42)),
        "--test-ratio", str(split.get("test_ratio", 0.95)),
    ]
    split_cmd.append(
        "--single-patient-train"
        if split.get("single_patient_train", True)
        else "--ratio-split"
    )

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "Unet_train.py"),
        "--input", str(train_pkl),
        "--work-dir", str(train_dir),
        "--device", device,
        "--seed", str(training.get("seed", 333)),
        "--validation-fraction", str(training.get("validation_fraction", 0.15)),
        "--n-trials", str(training.get("n_trials", 200)),
        "--epochs", str(training.get("epochs", 100)),
        "--early-stop-patience", str(training.get("early_stop_patience", 15)),
        "--dice-threshold", str(training.get("dice_threshold", 0.95)),
    ]
    add_flag(train_cmd, "--no-amp", bool(training.get("disable_amp", False)))

    evaluate_cmd = [
        sys.executable,
        str(REPO_ROOT / "Unet_eval.py"),
        "--work-dir", str(train_dir),
        "--test-pkl", str(test_pkl),
        "--output-pkl", str(predictions),
        "--stats-txt", str(predictions.with_name("inference_stats.tsv")),
        "--batch-size", str(evaluation.get("batch_size", 32)),
        "--pad-mode", str(evaluation.get("pad_mode", "reflect")),
        "--device", device,
    ]

    stages = [
        Stage("prepare", prepare_cmd, (prepared_400, prepared_1500)),
        Stage("normalize", normalize_cmd, (normalized,)),
        Stage("teacher", teacher_cmd, (teacher_output,)),
        Stage("clean-teacher", clean_cmd, (cleaned,)),
        Stage("split", split_cmd, (train_pkl, test_pkl)),
        Stage(
            "train",
            train_cmd,
            (train_dir / "config_unet_v2.json", train_dir / "checkpoints" / "best_model.pth"),
        ),
        Stage("evaluate", evaluate_cmd, (predictions, train_dir / "evaluation_results.json")),
    ]

    morphology_input = predictions
    if evaluation.get("clean_predictions", False):
        clean_student_cmd = [
            sys.executable,
            str(REPO_ROOT / "Cleaning_student.py"),
            "--input", str(predictions),
            "--output", str(cleaned_predictions),
            "--dataset", dataset,
            "--input-key", "Unet_preds",
        ]
        stages.append(Stage("clean-student", clean_student_cmd, (cleaned_predictions,)))
        morphology_input = cleaned_predictions

    morphology_cmd = [
        sys.executable,
        str(REPO_ROOT / "Morphology_extraction.py"),
        "--input", str(morphology_input),
        "--output-dir", str(morphology_dir),
        "--final-db", str(morphology_db),
        "--cell-stats", str(morphology_dir / "cell_stats.tsv"),
        "--area-scale", str(morphology.get("area_scale", 0.25)),
        "--model-keys",
        *[str(key) for key in morphology.get("model_keys", ["Ground_truth", "Unet_preds"])],
    ]
    add_flag(morphology_cmd, "--save-plots", bool(morphology.get("save_plots", False)))
    stages.append(Stage("morphology", morphology_cmd, (morphology_db,)))

    if classification.get("enabled", True):
        classifier_cmd = [
            sys.executable,
            str(REPO_ROOT / "Classifier.py"),
            "--input", str(morphology_db),
            "--output-dir", str(classification_dir),
            "--selected-model", str(classification.get("selected_model", "Unet_preds")),
            "--random-state", str(classification.get("random_state", 42)),
            "--outer-folds", str(classification.get("outer_folds", 3)),
            "--inner-folds", str(classification.get("inner_folds", 2)),
            "--k-values",
            *[str(value) for value in classification.get("k_values", [5, 10, 20, "all"])],
        ]
        stages.append(
            Stage("classify", classifier_cmd, (classification_dir / "summary_report.txt",))
        )
    return stages


def stages_for_command(stages: list[Stage], command: str) -> list[Stage]:
    if command == "all":
        return stages
    if command == "prepare":
        return [stage for stage in stages if stage.name in {
            "prepare", "normalize", "teacher", "clean-teacher", "split"
        }]
    if command == "train":
        return [stage for stage in stages if stage.name == "train"]
    if command == "evaluate":
        return [stage for stage in stages if stage.name in {
            "evaluate", "clean-student", "morphology", "classify"
        }]
    return []


def slice_stages(stages: list[Stage], start: str | None, end: str | None) -> list[Stage]:
    names = [stage.name for stage in stages]
    start_index = 0
    end_index = len(stages)
    if start:
        if start not in names:
            raise ValueError(f"Unknown --from-step '{start}'. Available: {', '.join(names)}")
        start_index = names.index(start)
    if end:
        if end not in names:
            raise ValueError(f"Unknown --to-step '{end}'. Available: {', '.join(names)}")
        end_index = names.index(end) + 1
    if start_index >= end_index:
        raise ValueError("--from-step must not come after --to-step.")
    return stages[start_index:end_index]


def outputs_exist(paths: Iterable[Path]) -> bool:
    paths = tuple(paths)
    return bool(paths) and all(path.exists() for path in paths)


def run_stages(stages: list[Stage], dry_run: bool, force: bool):
    if not stages:
        print("No stages selected.")
        return
    print("Selected stages: " + " -> ".join(stage.name for stage in stages))
    for index, stage in enumerate(stages, 1):
        command_text = shlex.join(stage.command)
        if outputs_exist(stage.outputs) and not force:
            print(f"[{index}/{len(stages)}] SKIP {stage.name}: outputs already exist")
            continue
        print(f"[{index}/{len(stages)}] RUN  {stage.name}")
        print(f"  {command_text}")
        if dry_run:
            continue
        try:
            subprocess.run(stage.command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Stage '{stage.name}' failed with exit code {exc.returncode}. "
                "Fix the reported error, then rerun; completed stages will be skipped."
            ) from exc
    print("Pipeline section complete.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, config_path = load_config(args.config)
    stages = make_stages(config, config_path)

    if args.command == "show-config":
        print(json.dumps(config, indent=2))
        print("\nResolved stages:")
        for stage in stages:
            print(f"- {stage.name}: {shlex.join(stage.command)}")
        return

    selected = stages_for_command(stages, args.command)
    selected = slice_stages(selected, args.from_step, args.to_step)
    run_stages(selected, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
