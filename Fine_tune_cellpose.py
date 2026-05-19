import os, random, pickle
import numpy as np
from skimage.measure import label
from skimage.util import img_as_float32
from cellpose import models
from cellpose import train as cp_train

# ─── 1. CONFIG ────────────────────────────────────────────
PATH_PKL   = '/mnt/lustre/home/claassen/clala950/DC-TSeg/Fine_tuned_cellpose/annotated_data_5_datasets.pkl'
SAVE_DIR   = '/mnt/lustre/home/claassen/clala950/DC-TSeg/Fine_tuned_cellpose/Cellpose_Fine_Tune_Models'
os.makedirs(SAVE_DIR, exist_ok=True)

PRETRAINED_MODEL = 'cyto2'

# ── SELECT DATASETS HERE ──────────────────────────────────
# use a subset:   DATASETS = ['eth400', 'eth1500']
# use one:        DATASETS = ['guck2025']
# use all:        DATASETS = None   ← will use every key in the pkl
DATASETS = None
# ─────────────────────────────────────────────────────────

# model name is built automatically from selected datasets
# ─── 2. LOAD DATA ─────────────────────────────────────────
db = pickle.load(open(PATH_PKL, 'rb'))

if DATASETS is None:
    DATASETS = sorted(db.keys())
    print(f"Using ALL datasets: {DATASETS}")
else:
    missing = [d for d in DATASETS if d not in db]
    if missing:
        raise KeyError(f"Datasets not found in pkl: {missing}. Available: {list(db.keys())}")
    print(f"Using datasets: {DATASETS}")

MODEL_NAME = f"{'_'.join(DATASETS)}_{PRETRAINED_MODEL}_finetuned"
print(f"Model name: {MODEL_NAME}")

# ─── 3. COLLECT IMAGES & MASKS FROM ALL SELECTED DATASETS ─
imgs, masks = [], []

for dataset in DATASETS:
    image_dict = db[dataset]['image']
    gt_dict    = db[dataset]['GT']

    common_keys = sorted(set(image_dict.keys()) & set(gt_dict.keys()))
    print(f"\n[{dataset}] Paired samples: {len(common_keys)}")

    count = 0
    for k in common_keys:
        img = img_as_float32(np.squeeze(image_dict[k]))
        msk = label(np.squeeze(gt_dict[k]) > 0).astype(np.int32)
        if msk.max() == 0:
            continue
        imgs.append(img)
        masks.append(msk)
        count += 1

    print(f"[{dataset}] After dropping empty masks: {count}")

print(f"\nTotal samples across all selected datasets: {len(imgs)}")

# ─── 4. TRAIN / VAL SPLIT ─────────────────────────────────
random.seed(0)
idx = list(range(len(imgs)))
random.shuffle(idx)
n_val = max(1, int(0.1 * len(imgs)))

X_train = [imgs[i][np.newaxis] for i in idx[n_val:]]
Y_train = [masks[i]            for i in idx[n_val:]]
X_val   = [imgs[i][np.newaxis] for i in idx[:n_val]]
Y_val   = [masks[i]            for i in idx[:n_val]]

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ─── 5. FINE-TUNE ─────────────────────────────────────────
base_model = models.CellposeModel(gpu=True, pretrained_model=PRETRAINED_MODEL)

model_path, _, _ = cp_train.train_seg(
    base_model.net,
    train_data=X_train,  train_labels=Y_train,
    test_data=X_val,     test_labels=Y_val,
    channels=[0, 0],     channel_axis=0,
    n_epochs=100,        learning_rate=1e-5,
    normalize=True,      compute_flows=True,
    rescale=True,
    min_train_masks=1,
    save_path=SAVE_DIR,
    model_name=MODEL_NAME
)

print("Done! Model saved at:", model_path)
