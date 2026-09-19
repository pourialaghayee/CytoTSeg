#!/usr/bin/env python3
"""
U-Net Optuna Multi-Objective Search Pipeline
- Single unified study: architecture + hyperparameters + augmentation searched jointly
- Two objectives: maximize Dice, minimize model parameters (Pareto front)
- Selection: 95% Dice threshold filter → smallest passing model
- Global trial budget: n_trials_total caps across ALL runs (resumable)
- Full plotting suite + optuna_trials.csv for paper figures
- Compatible output format with Unet_check.py
"""

import os
import pickle
import random
import math
import json
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch import amp
import optuna
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_optimization_history,
    plot_contour,
    plot_parallel_coordinate,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

PKL_PATH = ""
WORK_DIR = ""

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 333
USE_AMP  = torch.cuda.is_available()
VAL_FRAC = 0.15

OPTUNA_CONFIG = {
    # Search space — discrete
    'depths':        [1, 2, 3, 4],
    'bases':         [1, 2, 4, 8, 12, 16, 24, 32],
    'batch_sizes':   [32, 64, 128, 256, 512],
    'augmentations': ['minimal', 'light', 'light_strict'],
    # Search space — continuous
    'lr_min':           5e-5,
    'lr_max':           2e-4,
    'pos_weight_min':   0.4,
    'pos_weight_max':   0.8,
    'weight_decay_min': 1e-5,
    'weight_decay_max': 1e-3,
    # Training budget per trial
    'epochs':              100,
    'early_stop_patience': 15,
    'lr_floor':            1e-6,
    # Study budget — GLOBAL cap across all runs
    # If you resume, only the remaining trials are run
    'n_trials_total': 200,
    # Final model selection
    'dice_threshold_pct': 0.95,
}

log_file = None

def log_print(msg, also_print=True):
    if log_file is None:
        raise RuntimeError("Training runtime is not configured. Call configure_runtime() first.")
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
    if also_print:
        print(msg)

def configure_runtime(args):
    global PKL_PATH, WORK_DIR, DEVICE, USE_AMP, SEED, VAL_FRAC, log_file

    PKL_PATH = str(args.input.expanduser().resolve())
    WORK_DIR = str(args.work_dir.expanduser().resolve())
    if not Path(PKL_PATH).is_file():
        raise FileNotFoundError(f"Training pickle not found: {PKL_PATH}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    if args.device == "cpu":
        DEVICE = torch.device("cpu")
    elif args.device == "cuda":
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = args.seed
    VAL_FRAC = args.validation_fraction
    USE_AMP = DEVICE.type == "cuda" and not args.no_amp
    OPTUNA_CONFIG["n_trials_total"] = args.n_trials
    OPTUNA_CONFIG["epochs"] = args.epochs
    OPTUNA_CONFIG["early_stop_patience"] = args.early_stop_patience
    OPTUNA_CONFIG["dice_threshold_pct"] = args.dice_threshold

    for sub in ["", "checkpoints", "plots", "logs", "optuna_plots"]:
        Path(WORK_DIR, sub).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    log_file = os.path.join(
        WORK_DIR,
        "logs",
        f"optuna_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    log_print("=" * 80)
    log_print("U-NET OPTUNA MULTI-OBJECTIVE SEARCH PIPELINE")
    log_print("=" * 80)
    log_print(f"Device: {DEVICE}")
    log_print(f"Working directory: {WORK_DIR}")

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class CosineAnnealingSchedule:
    def __init__(self, optimizer, max_epochs, start_lr, min_lr=1e-7):
        self.optimizer  = optimizer
        self.max_epochs = max_epochs
        self.start_lr   = start_lr
        self.min_lr     = min_lr

    def step(self, epoch):
        progress = epoch / self.max_epochs
        lr = self.min_lr + (self.start_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience   = patience
        self.counter    = 0
        self.best_score = None

    def __call__(self, val_metric: float) -> bool:
        if self.best_score is None or val_metric > self.best_score:
            self.best_score = val_metric
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def pad_to_multiple(img: np.ndarray, multiple: int = 4, value: float = 0.0) -> np.ndarray:
    H, W  = img.shape
    H_new = int(math.ceil(H / multiple) * multiple)
    W_new = int(math.ceil(W / multiple) * multiple)
    return np.pad(img, ((0, H_new - H), (0, W_new - W)),
                  mode='constant', constant_values=value)

def binarize_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    return (m > 127.5).astype(np.float32) if m.max() > 1.0 else (m > 0.5).astype(np.float32)

# ============================================================================
# DATASET
# ============================================================================

class CellDataset(Dataset):
    def __init__(self, pkl_path: str, augmentation: str = 'minimal', verbose: bool = False):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.keys         = list(self.data.keys())
        self.augmentation = augmentation
        if verbose:
            log_print(f"Loaded {len(self.keys)} samples, aug='{augmentation}'")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        sample    = self.data[self.keys[idx]]
        img       = sample['image'].astype(np.float32)
        msk       = sample['Ground_truth'].astype(np.float32)
        H, W      = img.shape
        msk       = binarize_mask(msk)
        pad_value = float(np.median(img))
        img       = pad_to_multiple(img, 4, pad_value)
        msk       = pad_to_multiple(msk, 4, 0.0)

        if self.augmentation == 'minimal':
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                msk = np.flip(msk, axis=1).copy()

        elif self.augmentation in ('light', 'light_strict'):
            tight = self.augmentation == 'light_strict'
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy(); msk = np.flip(msk, axis=1).copy()
            if random.random() < 0.5:
                img = np.flip(img, axis=0).copy(); msk = np.flip(msk, axis=0).copy()
            if random.random() < 0.5:
                from scipy.ndimage import rotate
                angle = random.uniform(-10 if tight else -15, 10 if tight else 15)
                img = rotate(img, angle, reshape=False, order=1, mode='constant', cval=pad_value)
                msk = rotate(msk, angle, reshape=False, order=0, mode='constant', cval=0.0)
            lo, hi = (0.90, 1.10) if tight else (0.85, 1.15)
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(lo, hi), 0, 1)
            if not tight and random.random() < 0.5:
                c   = random.uniform(lo, hi)
                img = np.clip((img - img.mean()) * c + img.mean(), 0, 1)

        img = np.ascontiguousarray(img, dtype=np.float32)
        msk = np.ascontiguousarray(msk, dtype=np.float32)
        return torch.from_numpy(img)[None], torch.from_numpy(msk)[None], H, W


def pad_collate(batch):
    H_max = max(x.shape[1] for x, *_ in batch)
    W_max = max(x.shape[2] for x, *_ in batch)
    imgs, msks, sizes = [], [], []
    for img, msk, H, W in batch:
        pad = (0, W_max - img.shape[2], 0, H_max - img.shape[1])
        imgs.append(F.pad(img, pad, value=float(img.median())))
        msks.append(F.pad(msk, pad, value=0.0))
        sizes.append((H, W))
    return torch.stack(imgs), torch.stack(msks), sizes


def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# ============================================================================
# MODEL
# ============================================================================

def gn_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DynamicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16, depth=2):
        super().__init__()
        self.depth = depth
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        c_in, c_out = in_channels, base
        for _ in range(depth):
            self.downs.append(DoubleConv(c_in, c_out))
            self.pools.append(nn.MaxPool2d(2))
            c_in, c_out = c_out, c_out * 2
        self.bottleneck = DoubleConv(c_in, c_out)
        c_in = c_out
        self.ups, self.convs = nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            c_out = c_in // 2
            self.ups.append(nn.ConvTranspose2d(c_in, c_out, 2, 2))
            self.convs.append(DoubleConv(c_in, c_out))
            c_in = c_out
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.downs[i](x); skips.append(x); x = self.pools[i](x)
        x = self.bottleneck(x)
        for i, skip in enumerate(reversed(skips)):
            x = self.ups[i](x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = self.convs[i](torch.cat([x, skip], dim=1))
        return self.out(x)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# LOSS & METRICS
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits).view(logits.size(0), -1)
        t     = targets.view(targets.size(0), -1)
        inter = (p * t).sum(dim=1)
        return 1.0 - ((2.0 * inter + self.eps) / (p.sum(1) + t.sum(1) + self.eps)).mean()


def weighted_bce_dice_loss(logits, targets, pos_weight: float):
    bce  = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor([pos_weight], device=logits.device))
    dice = DiceLoss()(logits, targets)
    return 0.5 * bce + 0.5 * dice


@torch.no_grad()
def compute_metrics(logits, targets, sizes, threshold=0.5):
    probs = torch.sigmoid(logits).float()
    dices, ious = [], []
    for i, (H, W) in enumerate(sizes):
        pred  = (probs[i, 0, :H, :W] > threshold).float()
        gt    = targets[i, 0, :H, :W].float()
        inter = (pred * gt).sum()
        d = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        u = (inter + 1e-6) / (pred.sum() + gt.sum() - inter + 1e-6)
        dices.append(d.item()); ious.append(u.item())
    return float(np.mean(dices)), float(np.mean(ious))


def calculate_pos_weight(dataset, sample_size=1000):
    indices   = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    fg, total = 0, 0
    for idx in indices:
        _, msk, _, _ = dataset[int(idx)]
        fg += msk.sum().item(); total += msk.numel()
    fg_ratio = fg / total
    return float((1 - fg_ratio) / (fg_ratio + 1e-8)), float(fg_ratio)

# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def train_one_epoch(model, loader, optimizer, scaler, pos_weight):
    model.train()
    total = 0.0
    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu', enabled=USE_AMP):
            loss = weighted_bce_dice_loss(model(imgs), msks, pos_weight)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, pos_weight, threshold=0.5):
    model.eval()
    total, dices, ious = 0.0, [], []
    for imgs, msks, sizes in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        logits     = model(imgs)
        total     += weighted_bce_dice_loss(logits, msks, pos_weight).item() * imgs.size(0)
        d, u       = compute_metrics(logits, msks, sizes, threshold)
        dices.append(d); ious.append(u)
    return total / len(loader.dataset), float(np.mean(dices)), float(np.mean(ious))


@torch.no_grad()
def measure_inference_time(model, input_shape=(1, 1, 256, 256), n_warmup=5, n_runs=20):
    model.eval()
    x = torch.randn(input_shape, device=DEVICE)
    for _ in range(n_warmup): model(x)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs): model(x)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    return (time.time() - t0) / n_runs * 1000.0

# ============================================================================
# OPTUNA OBJECTIVE
# note: trial.report() / trial.should_prune() are NOT supported in
#       multi-objective studies — early stopping handles bad trials instead
# ============================================================================

_TRAIN_IDX  = None
_VAL_IDX    = None
_BASE_POS_W = None

def objective(trial: optuna.Trial):
    depth        = trial.suggest_categorical('depth',        OPTUNA_CONFIG['depths'])
    base         = trial.suggest_categorical('base',         OPTUNA_CONFIG['bases'])
    batch_size   = trial.suggest_categorical('batch_size',   OPTUNA_CONFIG['batch_sizes'])
    augment      = trial.suggest_categorical('augmentation', OPTUNA_CONFIG['augmentations'])
    lr_start     = trial.suggest_float('lr_start',
                                       OPTUNA_CONFIG['lr_min'],
                                       OPTUNA_CONFIG['lr_max'], log=True)
    pos_scale    = trial.suggest_float('pos_weight_scale',
                                       OPTUNA_CONFIG['pos_weight_min'],
                                       OPTUNA_CONFIG['pos_weight_max'])
    weight_decay = trial.suggest_float('weight_decay',
                                       OPTUNA_CONFIG['weight_decay_min'],
                                       OPTUNA_CONFIG['weight_decay_max'], log=True)

    pos_weight = _BASE_POS_W * pos_scale

    try:
        model    = DynamicUNet(1, 1, base=base, depth=depth).to(DEVICE)
        n_params = count_parameters(model)
        trial.set_user_attr('n_params', n_params)

        train_ds = CellDataset(PKL_PATH, augmentation=augment)
        val_ds   = CellDataset(PKL_PATH, augmentation='minimal')

        train_loader = DataLoader(
            Subset(train_ds, _TRAIN_IDX), batch_size=batch_size,
            shuffle=True,  num_workers=4, pin_memory=True,
            collate_fn=pad_collate, worker_init_fn=seed_worker)
        val_loader = DataLoader(
            Subset(val_ds, _VAL_IDX), batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=pad_collate, worker_init_fn=seed_worker)

        optimizer  = torch.optim.Adam(model.parameters(),
                                      lr=lr_start, weight_decay=weight_decay)
        scheduler  = CosineAnnealingSchedule(optimizer,
                                             max_epochs=OPTUNA_CONFIG['epochs'],
                                             start_lr=lr_start,
                                             min_lr=OPTUNA_CONFIG['lr_floor'])
        scaler     = amp.GradScaler(enabled=USE_AMP)
        early_stop = EarlyStopping(patience=OPTUNA_CONFIG['early_stop_patience'])

        best_dice, best_epoch = -1.0, 0
        history = []

        for epoch in range(1, OPTUNA_CONFIG['epochs'] + 1):
            scheduler.step(epoch - 1)
            train_loss               = train_one_epoch(model, train_loader, optimizer,
                                                       scaler, pos_weight)
            val_loss, val_dice, val_iou = validate(model, val_loader, pos_weight)

            history.append({'epoch': epoch, 'train_loss': train_loss,
                            'val_loss': val_loss, 'val_dice': val_dice,
                            'val_iou': val_iou})

            if val_dice > best_dice:
                best_dice, best_epoch = val_dice, epoch

            if early_stop(val_dice):
                break

        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('best_dice',  best_dice)
        trial.set_user_attr('history',    history)

        ckpt_path = os.path.join(WORK_DIR, 'checkpoints', f'trial_{trial.number:04d}.pth')
        torch.save({'model':        model.state_dict(),
                    'trial_number': trial.number,
                    'val_dice':     best_dice,
                    'n_params':     n_params,
                    'params':       dict(trial.params)}, ckpt_path)
        trial.set_user_attr('checkpoint', ckpt_path)

        log_print(f"Trial {trial.number:3d} | "
                  f"depth={depth} base={base:2d} bs={batch_size:3d} "
                  f"lr={lr_start:.2e} wd={weight_decay:.2e} "
                  f"pos={pos_scale:.2f} aug={augment:12s} | "
                  f"Dice={best_dice:.4f}  Params={n_params:,}")

        return best_dice, n_params

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            log_print(f"  ⚠ Trial {trial.number} OOM: depth={depth}, base={base}, bs={batch_size}")
            raise optuna.exceptions.TrialPruned()
        raise

# ============================================================================
# PARETO SELECTION
# ============================================================================

def select_best_trial(study: optuna.Study, dice_threshold_pct: float = 0.95):
    pareto_trials = study.best_trials
    if not pareto_trials:
        pareto_trials = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]

    best_dice = max(t.values[0] for t in pareto_trials)
    threshold = dice_threshold_pct * best_dice
    candidates = [t for t in pareto_trials if t.values[0] >= threshold]
    candidates.sort(key=lambda t: (t.values[1], -t.values[0]))
    chosen = candidates[0]

    log_print(f"\n{'='*60}")
    log_print(f"PARETO SELECTION RESULT")
    log_print(f"{'='*60}")
    log_print(f"  Pareto front size:          {len(pareto_trials)} trials")
    log_print(f"  Best Dice on front:         {best_dice:.4f}")
    log_print(f"  Dice threshold ({dice_threshold_pct*100:.0f}%):       {threshold:.4f}")
    log_print(f"  Candidates above threshold: {len(candidates)}")
    log_print(f"  ✓ Chosen trial #{chosen.number}")
    log_print(f"    Dice         = {chosen.values[0]:.4f}")
    log_print(f"    Params       = {chosen.values[1]:,}")
    log_print(f"    weight_decay = {chosen.params.get('weight_decay'):.2e}")
    log_print(f"    Config       = {chosen.params}")

    pure_best = max(pareto_trials, key=lambda t: t.values[0])
    if pure_best.number != chosen.number:
        log_print(f"\n  ℹ️  Pure-Dice winner: trial #{pure_best.number} | "
                  f"Dice={pure_best.values[0]:.4f} | Params={pure_best.values[1]:,}")
        log_print(f"     Pareto selection chose a smaller, efficient model instead.")

    return chosen

# ============================================================================
# FINAL TRAINING
# ============================================================================

def run_final_training(chosen_trial: optuna.trial.FrozenTrial, base_pos_weight: float):
    log_print("\n" + "=" * 80)
    log_print("FINAL TRAINING & THRESHOLD CALIBRATION")
    log_print("=" * 80)

    p            = chosen_trial.params
    depth        = p['depth']
    base         = p['base']
    batch_size   = p['batch_size']
    lr_start     = p['lr_start']
    pos_scale    = p['pos_weight_scale']
    augment      = p['augmentation']
    weight_decay = p['weight_decay']
    pos_weight   = base_pos_weight * pos_scale

    log_print(f"Final config: depth={depth}, base={base}, bs={batch_size}, "
              f"lr={lr_start:.2e}, wd={weight_decay:.2e}, "
              f"pos_scale={pos_scale:.2f}, aug={augment}")

    train_ds = CellDataset(PKL_PATH, augmentation=augment,   verbose=True)
    val_ds   = CellDataset(PKL_PATH, augmentation='minimal', verbose=False)
    N        = len(train_ds)
    if N < 2:
        raise ValueError("At least two training samples are required.")
    idxs     = list(range(N)); random.shuffle(idxs)
    n_val    = max(1, int(round(N * VAL_FRAC)))
    n_val    = min(n_val, N - 1)
    split    = N - n_val
    train_idx, val_idx = idxs[:split], idxs[split:]
    log_print(f"Data: {len(train_idx)} train / {len(val_idx)} val")

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=batch_size,
        shuffle=True,  num_workers=4, pin_memory=True,
        collate_fn=pad_collate, worker_init_fn=seed_worker)
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=pad_collate, worker_init_fn=seed_worker)

    model    = DynamicUNet(1, 1, base=base, depth=depth).to(DEVICE)
    n_params = count_parameters(model)
    log_print(f"Model parameters: {n_params:,}")

    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=lr_start, weight_decay=weight_decay)
    max_epochs = OPTUNA_CONFIG['epochs']
    patience   = OPTUNA_CONFIG['early_stop_patience']
    scheduler  = CosineAnnealingSchedule(optimizer, max_epochs, lr_start, OPTUNA_CONFIG['lr_floor'])
    scaler     = amp.GradScaler(enabled=USE_AMP)
    early_stop = EarlyStopping(patience=patience)

    best_dice, best_epoch = -1.0, 0
    history   = []
    ckpt_path = os.path.join(WORK_DIR, 'checkpoints', 'best_model.pth')

    for epoch in range(1, max_epochs + 1):
        lr         = scheduler.step(epoch - 1)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, pos_weight)
        val_loss, val_dice, val_iou = validate(model, val_loader, pos_weight)

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_dice': val_dice,
                        'val_iou': val_iou, 'lr': lr})

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({
                'model': model.state_dict(),
                'meta': {
                    'epoch': epoch, 'val_dice': val_dice, 'val_iou': val_iou,
                    'pos_weight': pos_weight,
                    'architecture': {'depth': depth, 'base': base},
                    'hyperparameters': {'lr_start': lr_start, 'batch_size': batch_size,
                                        'pos_weight_scale': pos_scale,
                                        'weight_decay': weight_decay},
                    'augmentation': augment,
                }}, ckpt_path)

        if epoch % 10 == 0:
            log_print(f"  Epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} "
                      f"dice={val_dice:.4f}  best={best_dice:.4f}@{best_epoch}")

        if early_stop(val_dice):
            log_print(f"  Early stopped at epoch {epoch}")
            break

    log_print(f"\n✓ Best Dice: {best_dice:.4f} at epoch {best_epoch}")

    # ── Threshold calibration ─────────────────────────────────────────────────
    log_print("Calibrating threshold...")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state['model'])
    model.eval()

    best_thr, best_thr_dice = 0.5, -1.0
    for thr in np.arange(0.3, 0.71, 0.02):
        dices = []
        for imgs, msks, sizes in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            d, _ = compute_metrics(model(imgs), msks, sizes, float(thr))
            dices.append(d)
        md = float(np.mean(dices))
        if md > best_thr_dice:
            best_thr_dice, best_thr = md, float(thr)
    log_print(f"✓ Best threshold: {best_thr:.3f}  (Dice={best_thr_dice:.4f})")

    inf_time = measure_inference_time(model)
    log_print(f"Inference: {inf_time:.2f} ms/image")

    config = {
        'model': {
            'architecture': 'DynamicUNet', 'base': int(base), 'depth': int(depth),
            'in_channels': 1, 'out_channels': 1, 'params': int(n_params),
        },
        'training': {
            'lr_start': float(lr_start), 'lr_min': float(OPTUNA_CONFIG['lr_floor']),
            'scheduler': 'CosineAnnealing', 'batch_size': int(batch_size),
            'max_epochs': int(max_epochs), 'early_stop_patience': int(patience),
            'best_epoch': int(best_epoch), 'epochs_trained': len(history),
            'loss_function': 'weighted_bce_dice', 'pos_weight': float(pos_weight),
            'weight_decay': float(weight_decay),
        },
        'preprocessing': {'norm_mode': 'pre_normalized', 'pad_multiple': 4},
        'threshold': float(best_thr),
        'validation': {'dice': float(best_dice), 'val_fraction': float(VAL_FRAC)},
        'augmentation': {'type': augment},
        'efficiency': {'inference_ms_per_image': float(inf_time)},
        'selection': {
            'method': 'pareto_multi_objective',
            'dice_threshold_pct': OPTUNA_CONFIG['dice_threshold_pct'],
            'optuna_trial': chosen_trial.number,
        },
        'checkpoint': 'best_model.pth',
        'checkpoint_full_path': ckpt_path,
    }

    cfg_path = os.path.join(WORK_DIR, 'config_unet_v2.json')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=2)
    log_print(f"✓ Config saved: {cfg_path}")

    return config, history

# ============================================================================
# PLOTTING SUITE
# ============================================================================

def plot_final_training_curves(history, best_epoch):
    epochs     = [h['epoch']      for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]
    val_dice   = [h['val_dice']   for h in history]
    val_iou    = [h['val_iou']    for h in history]
    lrs        = [h['lr']         for h in history]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].plot(epochs, train_loss, label='Train', marker='o', markersize=2)
    axes[0].plot(epochs, val_loss,   label='Val',   marker='s', markersize=2)
    axes[0].axvline(best_epoch, color='green', ls='--', alpha=0.6, label='Best')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_dice, color='green', marker='o', markersize=2, label='Val Dice')
    axes[1].axvline(best_epoch, color='green', ls='--', alpha=0.6)
    axes[1].set(title='Validation Dice', xlabel='Epoch', ylabel='Dice')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, val_iou, color='orange', marker='o', markersize=2, label='Val IoU')
    axes[2].axvline(best_epoch, color='green', ls='--', alpha=0.6)
    axes[2].set(title='Validation IoU', xlabel='Epoch', ylabel='IoU')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, lrs, color='purple', marker='o', markersize=2)
    axes[3].set(title='Learning Rate Schedule', xlabel='Epoch', ylabel='LR', yscale='log')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(WORK_DIR, 'plots', 'final_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log_print(f"✓ Training curves saved: {path}")


def plot_pareto_analysis(study: optuna.Study, chosen_trial: optuna.trial.FrozenTrial):
    complete   = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pareto     = study.best_trials

    all_dice   = [t.values[0] for t in complete]
    all_params = [t.values[1] for t in complete]
    all_nums   = [t.number    for t in complete]
    par_dice   = [t.values[0] for t in pareto]
    par_params = [t.values[1] for t in pareto]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc = axes[0].scatter(all_params, all_dice,
                         c=all_nums, cmap='viridis', alpha=0.5, s=40, label='All trials')
    axes[0].scatter(par_params, par_dice,
                    c='red', s=80, zorder=5, marker='D', label='Pareto front')
    axes[0].scatter(chosen_trial.values[1], chosen_trial.values[0],
                    c='lime', s=200, zorder=6, marker='*',
                    label=f'Chosen (#{chosen_trial.number})')
    pure_best = max(pareto, key=lambda t: t.values[0])
    if pure_best.number != chosen_trial.number:
        axes[0].scatter(pure_best.values[1], pure_best.values[0],
                        c='cyan', s=150, zorder=6, marker='^',
                        label=f'Pure-Dice best (#{pure_best.number})')
    plt.colorbar(sc, ax=axes[0], label='Trial number')
    axes[0].set(title='Pareto Front: Dice vs Parameters',
                xlabel='Parameters', ylabel='Dice Score')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    par_sorted = sorted(pareto, key=lambda t: t.values[1])
    xs = [t.values[1] for t in par_sorted]
    ys = [t.values[0] for t in par_sorted]
    axes[1].step([xs[0]] + xs, [ys[0]] + ys, where='post',
                 color='red', alpha=0.4, linewidth=1.5)
    axes[1].scatter(xs, ys, c='red', s=80, zorder=5)
    for t in par_sorted:
        d = t.params.get('depth', '?')
        b = t.params.get('base',  '?')
        axes[1].annotate(f"d{d}b{b}", (t.values[1], t.values[0]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[1].scatter(chosen_trial.values[1], chosen_trial.values[0],
                    c='lime', s=200, zorder=6, marker='*', label='Chosen')
    axes[1].set(title='Pareto Front (annotated)',
                xlabel='Parameters', ylabel='Dice Score')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(WORK_DIR, 'optuna_plots', 'pareto_front.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log_print(f"✓ Pareto plot saved: {path}")


def plot_optuna_diagnostics(study: optuna.Study):
    plots_dir = os.path.join(WORK_DIR, 'optuna_plots')

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_optimization_history(study, target=lambda t: t.values[0],
                                  target_name="Dice", ax=ax)
        ax.set_title("Optuna: Dice Score Across Trials")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'optuna_history_dice.png'), dpi=150)
        plt.close()
        log_print("✓ Optimization history saved")
    except Exception as e:
        log_print(f"  ⚠ history plot skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_param_importances(study, target=lambda t: t.values[0],
                               target_name="Dice", ax=ax)
        ax.set_title("Hyperparameter Importance → Dice Score")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'param_importance_dice.png'), dpi=150)
        plt.close()
        log_print("✓ Param importance (Dice) saved")
    except Exception as e:
        log_print(f"  ⚠ importance (Dice) skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_param_importances(study, target=lambda t: t.values[1],
                               target_name="n_params", ax=ax)
        ax.set_title("Hyperparameter Importance → Model Size")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'param_importance_size.png'), dpi=150)
        plt.close()
        log_print("✓ Param importance (size) saved")
    except Exception as e:
        log_print(f"  ⚠ importance (size) skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_contour(study, params=['lr_start', 'base'],
                     target=lambda t: t.values[0],
                     target_name="Dice", ax=ax)
        ax.set_title("Contour: LR × Base → Dice")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'contour_lr_base.png'), dpi=150)
        plt.close()
        log_print("✓ Contour (LR × base) saved")
    except Exception as e:
        log_print(f"  ⚠ contour skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_parallel_coordinate(
            study,
            params=['depth', 'base', 'lr_start', 'weight_decay',
                    'batch_size', 'pos_weight_scale'],
            target=lambda t: t.values[0],
            target_name="Dice", ax=ax)
        ax.set_title("Parallel Coordinates: Hyperparameters vs Dice")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parallel_coordinates.png'), dpi=150)
        plt.close()
        log_print("✓ Parallel coordinates saved")
    except Exception as e:
        log_print(f"  ⚠ parallel coords skipped: {e}")

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_contour(study, params=['weight_decay', 'lr_start'],
                     target=lambda t: t.values[0],
                     target_name="Dice", ax=ax)
        ax.set_title("Contour: Weight Decay × LR → Dice")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'contour_wd_lr.png'), dpi=150)
        plt.close()
        log_print("✓ Contour (WD × LR) saved")
    except Exception as e:
        log_print(f"  ⚠ contour (WD × LR) skipped: {e}")

    complete  = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    aug_types = OPTUNA_CONFIG['augmentations']
    colors    = ['tab:blue', 'tab:orange', 'tab:green']
    fig, ax   = plt.subplots(figsize=(9, 6))
    for aug, col in zip(aug_types, colors):
        ts = [t for t in complete if t.params.get('augmentation') == aug]
        if ts:
            ax.scatter([t.values[1] for t in ts], [t.values[0] for t in ts],
                       label=aug, color=col, alpha=0.6, s=40)
    ax.set(title='Dice vs Params by Augmentation Type',
           xlabel='Parameters', ylabel='Dice Score')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'augmentation_comparison.png'), dpi=150)
    plt.close()
    log_print("✓ Augmentation comparison saved")

    if complete:
        wds         = [t.params.get('weight_decay') for t in complete]
        dices       = [t.values[0] for t in complete]
        wds_c, dc_c = zip(*[(w, d) for w, d in zip(wds, dices) if w is not None])
        fig, ax     = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(wds_c, dc_c, c=dc_c, cmap='RdYlGn', alpha=0.7, s=50)
        plt.colorbar(sc, ax=ax, label='Dice')
        ax.set(title='Weight Decay vs Dice Score',
               xlabel='Weight Decay', ylabel='Dice', xscale='log')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'weight_decay_vs_dice.png'), dpi=150)
        plt.close()
        log_print("✓ Weight decay vs Dice saved")


def save_trials_csv(study: optuna.Study):
    """
    Save ALL trial info to CSV — use this file for paper plots later.
    Columns: trial, dice, n_params, on_pareto, depth, base, batch_size,
             lr_start, weight_decay, pos_weight_scale, augmentation,
             best_epoch, checkpoint
    """
    rows = []
    pareto_numbers = {p.number for p in study.best_trials}
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            'trial':            t.number,
            'dice':             t.values[0],
            'n_params':         t.values[1],
            'on_pareto':        t.number in pareto_numbers,
            'best_epoch':       t.user_attrs.get('best_epoch', ''),
            'checkpoint':       t.user_attrs.get('checkpoint', ''),
        }
        row.update(t.params)
        rows.append(row)

    path = os.path.join(WORK_DIR, 'optuna_trials.csv')
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    log_print(f"✓ All trials CSV saved: {path}  ({len(rows)} trials)")
    log_print(f"  → Use this file with the standalone plotting script for paper figures")


def create_final_summary(study: optuna.Study,
                         chosen_trial: optuna.trial.FrozenTrial,
                         config: dict):
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    path     = os.path.join(WORK_DIR, 'final_summary.txt')

    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OPTUNA MULTI-OBJECTIVE U-NET SEARCH — FINAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total trials:         {len(study.trials)}\n")
        f.write(f"Completed:            {len(complete)}\n")
        f.write(f"Pruned (OOM):         {len(pruned)}\n")
        f.write(f"Pareto front size:    {len(study.best_trials)}\n")
        f.write(f"Global budget:        {OPTUNA_CONFIG['n_trials_total']}\n\n")

        f.write("CHOSEN MODEL\n" + "-" * 60 + "\n")
        f.write(f"  Trial #:      {chosen_trial.number}\n")
        f.write(f"  Dice:         {chosen_trial.values[0]:.4f}\n")
        f.write(f"  Params:       {chosen_trial.values[1]:,}\n")
        for k, v in chosen_trial.params.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nFINAL TRAINING\n" + "-" * 60 + "\n")
        f.write(f"  Validation Dice:   {config['validation']['dice']:.4f}\n")
        f.write(f"  Threshold:         {config['threshold']:.3f}\n")
        f.write(f"  Inference time:    {config['efficiency']['inference_ms_per_image']:.2f} ms\n")
        f.write(f"  Best epoch:        {config['training']['best_epoch']}\n")
        f.write(f"  Weight decay:      {config['training']['weight_decay']:.2e}\n\n")

        f.write("PARETO FRONT\n" + "-" * 60 + "\n")
        for t in sorted(study.best_trials, key=lambda x: x.values[1]):
            f.write(f"  Trial #{t.number:3d}: Dice={t.values[0]:.4f}  "
                    f"Params={t.values[1]:,}  "
                    f"depth={t.params.get('depth')} base={t.params.get('base')}  "
                    f"wd={t.params.get('weight_decay'):.2e}\n")

        f.write("\n" + "=" * 80 + "\nEND\n" + "=" * 80 + "\n")

    log_print(f"✓ Summary saved: {path}")

# ============================================================================
# MAIN
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Search for a compact U-Net and train the selected student model."
    )
    parser.add_argument("--input", type=Path, required=True, help="Flat training pickle.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--dice-threshold", type=float, default=0.95)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser


def main(argv=None):
    global _TRAIN_IDX, _VAL_IDX, _BASE_POS_W

    args = build_parser().parse_args(argv)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if args.n_trials < 1 or args.epochs < 1 or args.early_stop_patience < 1:
        raise ValueError("Trials, epochs, and early-stop patience must be positive.")
    if not 0.0 < args.dice_threshold <= 1.0:
        raise ValueError("--dice-threshold must be in (0, 1].")
    configure_runtime(args)

    start_time = time.time()

    log_print("Loading dataset for data split...")
    base_ds = CellDataset(PKL_PATH, augmentation='minimal', verbose=True)
    N       = len(base_ds)
    idxs    = list(range(N)); random.shuffle(idxs)
    n_val   = max(1, int(round(N * VAL_FRAC)))
    _TRAIN_IDX      = idxs[:N - n_val]
    _VAL_IDX        = idxs[N - n_val:]
    _BASE_POS_W, fg = calculate_pos_weight(Subset(base_ds, _TRAIN_IDX))
    log_print(f"Split: {len(_TRAIN_IDX)} train / {len(_VAL_IDX)} val | "
              f"fg_ratio={fg:.4f}, base_pos_weight={_BASE_POS_W:.3f}")

    sampler = optuna.samplers.NSGAIISampler(seed=SEED)

    study = optuna.create_study(
        study_name="unet_pareto_search",
        directions=["maximize", "minimize"],
        sampler=sampler,
        storage=f"sqlite:///{os.path.join(WORK_DIR, 'optuna_study.db')}",
        load_if_exists=True,
    )

    # ── Global trial budget ───────────────────────────────────────────────────
    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    remaining    = max(0, OPTUNA_CONFIG['n_trials_total'] - already_done)

    log_print(f"\nGlobal trial budget:       {OPTUNA_CONFIG['n_trials_total']}")
    log_print(f"Trials already completed:  {already_done}")
    log_print(f"Trials remaining this run: {remaining}")
    log_print("Objectives: maximize Dice  |  minimize n_params")
    log_print("Sampler: NSGA-II  |  Early stopping handles bad trials\n")

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        log_print("✓ Global trial budget already reached. Skipping optimization.")
        log_print("  Running final selection and training on existing results...")

    chosen_trial    = select_best_trial(study, OPTUNA_CONFIG['dice_threshold_pct'])
    config, history = run_final_training(chosen_trial, _BASE_POS_W)

    plot_final_training_curves(history, config['training']['best_epoch'])
    plot_pareto_analysis(study, chosen_trial)
    plot_optuna_diagnostics(study)
    save_trials_csv(study)
    create_final_summary(study, chosen_trial, config)

    elapsed = time.time() - start_time
    log_print("\n" + "=" * 80)
    log_print("PIPELINE COMPLETE!")
    log_print("=" * 80)
    log_print(f"Total time: {elapsed / 3600:.2f} hours")
    log_print(f"\nOutputs in: {WORK_DIR}")
    log_print(f"  best_model.pth               ← final trained model")
    log_print(f"  config_unet_v2.json          ← inference config")
    log_print(f"  optuna_study.db              ← full study (resumable)")
    log_print(f"  optuna_trials.csv            ← USE THIS for paper plots")
    log_print(f"  optuna_plots/                ← pareto, importances, contours")
    log_print(f"  plots/final_training_curves.png")
    log_print(f"  final_summary.txt")
    log_print(f"  logs/                        ← full run log")


if __name__ == "__main__":
    main()
