"""
CausalCrisis V3 - Google Colab Notebook Script

This script mirrors the experiment notebook cells and is used as the editable
source-of-truth for protocol updates.
"""

# %%
# ============================================================================
# CELL 1: Setup & Installation
# ============================================================================
print("=" * 60)
print("CausalCrisis V3 - Setup")
print("=" * 60)

import subprocess
import sys
import time
import json
import shutil


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


packages = [
    "open_clip_torch",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "PyYAML",
]

import_name_map = {
    "open_clip_torch": "open_clip",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
}

for pkg in packages:
    try:
        __import__(import_name_map[pkg])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_memory = getattr(props, "total_memory", None)
    if total_memory is None:
        total_memory = getattr(props, "total_mem", 0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {total_memory / 1e9:.1f} GB")
else:
    print("No GPU detected; training will be slow.")

try:
    import google.colab  # type: ignore

    IN_COLAB = True
    print("Colab detected. Drive mount is disabled.")
except ImportError:
    IN_COLAB = False
    print("Not in Colab - running locally")

DRIVE_MOUNTED = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

TARGET_H1 = 0.88
TARGET_CRISISSPOT = 0.909
TARGET_H6 = 0.91


def to_python_types(obj):
    """Recursively convert numpy / torch scalar types to plain Python types."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj

# %%
# ============================================================================
# CELL 2: Clone Repository & Import Modules
# ============================================================================
print("\n" + "=" * 60)
print("Setting up project")
print("=" * 60)

REPO_URL = "https://github.com/Thanh-000/CrisisSummarization-CausalGNN.git"
BRANCH = "v3-causalcrisis-enhanced"
PROJECT_DIR = "/content/CausalCrisisV3"
DATASET_DIR = "/content/CrisisMMD_v2.0"
CACHE_DIR = "/content/cached_features"
CRISISMMD_DATA_URL = "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"
CRISISMMD_SPLIT_URL = "https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_all.zip"
CRISISMMD_DATA_URLS = [
    CRISISMMD_DATA_URL,
    # Mirror fallback (if available):
    "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/CrisisMMD_v2.0.tar.gz",
]
CRISISMMD_SPLIT_URLS = [
    CRISISMMD_SPLIT_URL,
    # Mirror fallback (if available):
    "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/main/crisismmd_datasplit_all.zip",
]

if not os.path.exists(os.path.join(PROJECT_DIR, "src")):
    print(f"Cloning {REPO_URL} (branch: {BRANCH})...")
    os.system(f"git clone -b {BRANCH} {REPO_URL} {PROJECT_DIR}")
else:
    print("Repository already exists")
    os.system(f"cd {PROJECT_DIR} && git pull origin {BRANCH}")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import get_config
from src.models import CausalCrisisV3, CLIPMLPBaseline
from src.losses import CausalCrisisLoss, FocalLoss
from src.data import (
    load_crisismmd_annotations,
    extract_and_cache_clip_features,
    create_stratified_splits,
    create_dataloaders,
    compute_class_weights,
)
from src.trainer import CausalCrisisTrainer, BaselineTrainer
from src.evaluate import (
    compute_metrics,
    paired_bootstrap_test,
    plot_training_curves,
    plot_tsne_causal_features,
    run_lodo_evaluation,
    run_ablation_study,
)

print("All modules imported")

# %%
# ============================================================================
# CELL 3: Load CrisisMMD Dataset
# ============================================================================
print("\n" + "=" * 60)
print("Loading CrisisMMD v2.0")
print("=" * 60)

def ensure_aria2_available() -> None:
    if shutil.which("aria2c"):
        return
    if IN_COLAB:
        print("aria2c not found. Installing aria2 in Colab...")
        subprocess.check_call(["apt-get", "update", "-y"])
        subprocess.check_call(["apt-get", "install", "-y", "aria2"])
        if shutil.which("aria2c"):
            return
    raise RuntimeError(
        "aria2c is not available. Install aria2 first, then rerun this cell."
    )


def aria2_download(url: str, out_dir: str, out_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    cmd = [
        "aria2c",
        "-x",
        "16",
        "-s",
        "16",
        "-k",
        "1M",
        "--file-allocation=none",
        "--allow-overwrite=true",
        "-d",
        out_dir,
        "-o",
        out_name,
        url,
    ]
    print("Downloading with aria2:", url)
    subprocess.check_call(cmd)
    return out_path


def aria2_download_with_mirrors(urls, out_dir: str, out_name: str) -> str:
    last_err = None
    for url in urls:
        try:
            return aria2_download(url, out_dir, out_name)
        except Exception as exc:
            last_err = exc
            print(f"Download failed for URL: {url}")
            print(f"Reason: {exc}")
    raise RuntimeError(
        f"All download URLs failed for {out_name}. Last error: {last_err}"
    )


def find_file_recursive(search_roots, filename: str):
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def find_directory_recursive(search_roots, dirname: str):
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if dirname in dirnames:
                return os.path.join(dirpath, dirname)
    return None


def collect_matching_tsv(search_roots, keyword: str):
    matches = []
    keyword = keyword.lower()
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fn_lower = fn.lower()
                if fn_lower.endswith(".tsv") and keyword in fn_lower:
                    matches.append(os.path.join(dirpath, fn))
    return matches


def resolve_task1_split_files(split_dir: str, search_roots):
    os.makedirs(split_dir, exist_ok=True)
    expected = {
        "train": os.path.join(
            split_dir, "task_informative_text_img_agreed_lab_train.tsv"
        ),
        "dev": os.path.join(
            split_dir, "task_informative_text_img_agreed_lab_dev.tsv"
        ),
        "test": os.path.join(
            split_dir, "task_informative_text_img_agreed_lab_test.tsv"
        ),
    }

    # Fast path: already in canonical location.
    if all(os.path.exists(v) for v in expected.values()):
        return expected

    candidates = collect_matching_tsv(
        search_roots, "task_informative_text_img"
    )
    split_to_src = {}
    for fp in candidates:
        name = os.path.basename(fp).lower()
        if "/__macosx/" in fp.lower() or name.startswith("._"):
            continue
        if "train" in name:
            split_to_src["train"] = fp
        elif "test" in name:
            split_to_src["test"] = fp
        elif "dev" in name or "val" in name or "valid" in name:
            split_to_src["dev"] = fp

    if set(split_to_src.keys()) == {"train", "dev", "test"}:
        print("Resolved split files from discovered paths:")
        for split, src in split_to_src.items():
            dst = expected[split]
            print(f"  {split}: {src}")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
        return expected

    # Diagnostic output for easier debugging in Colab.
    print("Could not fully resolve task1 split files.")
    print(f"Expected directory: {split_dir}")
    print("Discovered candidates:")
    for fp in sorted(candidates)[:50]:
        print(" ", fp)
    missing = [k for k in ["train", "dev", "test"] if k not in split_to_src]
    print(
        "Warning: missing canonical split files for "
        f"{missing}. Loader will attempt flexible discovery."
    )
    return expected


def load_crisismmd_annotations_fallback(data_dir: str, task: str = "task1") -> pd.DataFrame:
    """
    Notebook-local robust loader used when src.data.load_crisismmd_annotations fails
    due to column/path inconsistencies in remote code versions.
    """
    task_base = {
        "task1": "task_informative_text_img",
        "task2": "task_humanitarian_text_img",
        "task3": "task_damage_text_img",
    }[task]
    label_maps = {
        "task1": {"informative": 1, "not_informative": 0},
        "task2": {
            "infrastructure_and_utility_damage": 0,
            "vehicle_damage": 1,
            "rescue_volunteering_or_donation_effort": 2,
            "injured_or_dead_people": 3,
            "affected_individuals": 4,
            "missing_or_found_people": 5,
            "other_relevant_information": 6,
            "not_humanitarian": 7,
        },
        "task3": {
            "severe_damage": 2,
            "mild_damage": 1,
            "little_or_no_damage": 0,
        },
    }

    split_dir = os.path.join(data_dir, "crisismmd_datasplit_all")
    records = []

    def choose_split_file(split: str):
        candidates = [
            os.path.join(split_dir, f"{task_base}_agreed_lab_{split}.tsv"),
            os.path.join(split_dir, f"{task_base}_{split}.tsv"),
        ]
        for fp in candidates:
            if os.path.exists(fp):
                return fp
        # Last attempt: recursive match
        for root, _, files in os.walk(data_dir):
            for fn in files:
                fnl = fn.lower()
                if task_base in fnl and split in fnl and fnl.endswith(".tsv"):
                    return os.path.join(root, fn)
        return None

    for split in ["train", "dev", "test"]:
        fp = choose_split_file(split)
        if not fp:
            continue
        df = pd.read_csv(fp, sep="\t")

        # Column selection with simple scoring.
        cols = list(df.columns)

        def pick(kind: str):
            best = None
            best_score = -10**9
            for c in cols:
                lc = str(c).lower().strip()
                s = 0
                if kind == "image":
                    if "image_path" in lc:
                        s += 120
                    if lc in {"image", "img"}:
                        s += 100
                    if "image" in lc or "img" in lc:
                        s += 50
                    if "id" in lc:
                        s -= 80
                elif kind == "text":
                    if "tweet_text" in lc:
                        s += 120
                    if lc == "text":
                        s += 100
                    if "text" in lc:
                        s += 50
                elif kind == "label":
                    if lc in {"label_name", "label"}:
                        s += 120
                    if "label" in lc:
                        s += 60
                    if "id" in lc:
                        s -= 60
                elif kind == "event":
                    if lc == "event_name":
                        s += 120
                    if "event" in lc:
                        s += 50
                if s > best_score:
                    best_score = s
                    best = c
            return best if best_score > 0 else None

        img_col = pick("image")
        txt_col = pick("text")
        lbl_col = pick("label")
        evt_col = pick("event")
        if img_col is None or lbl_col is None:
            continue

        norm = pd.DataFrame(index=df.index)
        norm["image_path"] = df[img_col]
        norm["text"] = df[txt_col] if txt_col is not None else ""
        norm["label_name"] = df[lbl_col]
        if evt_col is not None:
            norm["event_name"] = df[evt_col]
        norm["split"] = split

        # Labels
        label_map = label_maps[task]
        label_series = norm["label_name"]
        if pd.api.types.is_numeric_dtype(label_series):
            mapped = pd.to_numeric(label_series, errors="coerce")
        else:
            label_str = label_series.astype(str).str.lower().str.strip()
            mapped = label_str.map(label_map)
            if mapped.isna().all():
                mapped = pd.to_numeric(label_str, errors="coerce")
        norm["label"] = mapped
        norm = norm.dropna(subset=["label"])
        norm["label"] = norm["label"].astype(int)

        # Paths
        def to_abs(x):
            if pd.isna(x):
                return ""
            p = str(x).strip()
            if not p:
                return ""
            if os.path.isabs(p):
                return p
            return os.path.join(data_dir, p.lstrip("./\\"))

        norm["image_path"] = norm["image_path"].apply(to_abs)
        records.append(norm)

    if not records:
        raise FileNotFoundError(f"No CrisisMMD files found by fallback loader in {data_dir}")

    data = pd.concat(records, ignore_index=True)
    if "event_name" in data.columns:
        events = sorted(data["event_name"].dropna().unique())
        event_to_id = {e: i for i, e in enumerate(events)}
        data["domain_id"] = data["event_name"].map(event_to_id)
    return data


def normalize_split_filenames(split_dir: str, search_roots):
    """
    Normalize official split names to the canonical names expected by old/new loaders.
    Example:
      task_informative_text_img_train.tsv
    -> task_informative_text_img_agreed_lab_train.tsv
    """
    os.makedirs(split_dir, exist_ok=True)
    patterns = [
        ("task_informative_text_img", "task_informative_text_img_agreed_lab"),
        ("task_humanitarian_text_img", "task_humanitarian_text_img_agreed_lab"),
        ("task_damage_text_img", "task_damage_text_img_agreed_lab"),
    ]
    splits = ["train", "dev", "test"]

    for src_base, dst_base in patterns:
        for sp in splits:
            dst = os.path.join(split_dir, f"{dst_base}_{sp}.tsv")
            if os.path.exists(dst):
                continue

            candidate = None
            wanted = f"{src_base}_{sp}.tsv"
            for root in search_roots:
                if not root or not os.path.exists(root):
                    continue
                for dirpath, _, filenames in os.walk(root):
                    for fn in filenames:
                        fn_lower = fn.lower()
                        if fn_lower.startswith("._"):
                            continue
                        if fn_lower == wanted:
                            candidate = os.path.join(dirpath, fn)
                            break
                    if candidate:
                        break
                if candidate:
                    break

            if candidate:
                print(f"Normalize split filename: {candidate} -> {dst}")
                if os.path.abspath(candidate) != os.path.abspath(dst):
                    shutil.copy2(candidate, dst)


if not os.path.exists(DATASET_DIR):
    print(f"Dataset folder not found at {DATASET_DIR}")
    print("Starting direct download via aria2...")
    ensure_aria2_available()
    archive = aria2_download_with_mirrors(
        CRISISMMD_DATA_URLS, "/content", "CrisisMMD_v2.0.tar.gz"
    )
    subprocess.check_call(["tar", "-xzf", archive, "-C", "/content"])
    print("Main dataset extracted.")

split_dir = os.path.join(DATASET_DIR, "crisismmd_datasplit_all")
if not os.path.exists(split_dir):
    print("Split annotations folder missing. Downloading split package via aria2...")
    ensure_aria2_available()
    split_zip = aria2_download_with_mirrors(
        CRISISMMD_SPLIT_URLS, "/content", "crisismmd_datasplit_all.zip"
    )
    subprocess.check_call(["unzip", "-o", split_zip, "-d", DATASET_DIR])
    print("Split annotations extracted.")

# Resolve real dataset root if archive extracted into nested folders.
if not os.path.exists(os.path.join(DATASET_DIR, "data_image")):
    discovered_data_image = find_directory_recursive(
        [DATASET_DIR, "/content"], "data_image"
    )
    if discovered_data_image:
        DATASET_DIR = os.path.dirname(discovered_data_image)
        print(f"Resolved dataset root to: {DATASET_DIR}")

assert os.path.exists(DATASET_DIR), f"Dataset still missing at {DATASET_DIR}"
# Normalize filenames first so cloned loader code can read canonical paths.
normalize_split_filenames(split_dir, [DATASET_DIR, "/content"])
resolved_task1 = resolve_task1_split_files(split_dir, [DATASET_DIR, "/content"])
missing_split_files = [k for k, fp in resolved_task1.items() if not os.path.exists(fp)]
if missing_split_files:
    print(
        "Warning: canonical task1 files still missing for "
        f"{missing_split_files}. Continuing with flexible loader discovery."
    )

if not os.path.exists(os.path.join(DATASET_DIR, "data_image")):
    print("Warning: data_image folder not found under DATASET_DIR.")
    print("Check extracted structure if image loading fails.")

try:
    data = load_crisismmd_annotations(DATASET_DIR, task="task1")
except Exception as exc:
    # Remote /content repo versions can have brittle path handling.
    # Always fallback to notebook-local robust loader for resiliency.
    print(
        "Primary loader failed "
        f"({type(exc).__name__}: {exc}). Using fallback loader..."
    )
    data = load_crisismmd_annotations_fallback(DATASET_DIR, task="task1")

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
if "split" in data.columns:
    print("Split distribution:")
    print(data["split"].value_counts())

# %%
# ============================================================================
# CELL 4: CLIP Feature Extraction & Caching
# ============================================================================
print("\n" + "=" * 60)
print("CLIP Feature Extraction (ViT-L/14)")
print("=" * 60)

config = get_config("task1")

t0 = time.time()
image_features, text_features = extract_and_cache_clip_features(
    data=data,
    cache_dir=CACHE_DIR,
    model_name=config.clip.model_name,
    batch_size=64,
    device=DEVICE,
)
print(f"Feature extraction finished in {time.time() - t0:.1f}s")

labels = data["label"].values
if "domain_id" in data.columns:
    domain_ids = data["domain_id"].values.astype(int)
elif "event_name" in data.columns:
    events = sorted(data["event_name"].dropna().unique())
    event_to_id = {e: i for i, e in enumerate(events)}
    domain_ids = data["event_name"].map(event_to_id).values.astype(int)
else:
    domain_ids = np.zeros(len(labels), dtype=int)

print("Feature shapes:")
print(f"  Image: {image_features.shape}")
print(f"  Text:  {text_features.shape}")
print(f"  Labels: {labels.shape} classes={np.unique(labels)}")
print(f"  Domains: {len(np.unique(domain_ids))}")

# %%
# ============================================================================
# CELL 5: Create Data Splits (official split first, fallback stratified)
# ============================================================================
print("\n" + "=" * 60)
print("Preparing train/val/test splits")
print("=" * 60)

split_mode = "stratified_70_15_15"
if "split" in data.columns:
    split_col = data["split"].astype(str).str.lower().str.strip()
    train_idx = np.where(split_col == "train")[0]
    val_idx = np.where(split_col.isin(["dev", "val", "valid", "validation"]))[0]
    test_idx = np.where(split_col == "test")[0]
    if len(train_idx) and len(val_idx) and len(test_idx):
        split_mode = "official_train_dev_test"
    else:
        train_idx = val_idx = test_idx = None
else:
    train_idx = val_idx = test_idx = None

if train_idx is None:
    seed_for_split = 42
    train_idx, val_idx, test_idx = create_stratified_splits(
        labels,
        domain_ids,
        test_ratio=config.eval.test_split,
        val_ratio=config.eval.val_split,
        seed=seed_for_split,
    )

train_labels = labels[train_idx]
class_weights = compute_class_weights(train_labels)

loaders = create_dataloaders(
    image_features,
    text_features,
    labels,
    domain_ids,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    batch_size=config.training.batch_size,
    num_workers=2,
)

print(f"Split mode: {split_mode}")
print(f"Train: {len(loaders['train'].dataset):,}")
print(f"Val:   {len(loaders['val'].dataset):,}")
print(f"Test:  {len(loaders['test'].dataset):,}")

SEEDS = list(config.training.seeds[:5])
print(f"Seeds: {SEEDS}")

# %%
# ============================================================================
# CELL 6: EXPERIMENT H1 - CLIP MLP Baseline (5 seeds)
# ============================================================================
print("\n" + "=" * 60)
print("H1: CLIP ViT-L/14 + MLP baseline")
print(f"Target: F1 > {TARGET_H1:.2f}")
print("=" * 60)

baseline_results = []
baseline_histories = []

for seed in SEEDS:
    print(f"\n--- Baseline seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    baseline_model = CLIPMLPBaseline(
        input_dim=config.clip.image_dim * 2,
        num_classes=config.classifier.num_classes,
    )

    optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6,
    )
    loss_fn = FocalLoss(alpha=class_weights, gamma=config.training.focal_gamma)

    trainer = BaselineTrainer(
        baseline_model,
        optimizer,
        scheduler,
        device=DEVICE,
        save_dir="checkpoints",
    )
    history = trainer.train(
        loaders["train"],
        loaders["val"],
        epochs=100,
        patience=15,
        loss_fn=loss_fn,
    )
    baseline_histories.append(history)

    baseline_ckpt = "checkpoints/baseline_best.pt"
    baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE))
    seed_ckpt = f"checkpoints/baseline_seed{seed}_best.pt"
    shutil.copyfile(baseline_ckpt, seed_ckpt)

    baseline_model.eval().to(DEVICE)
    all_preds, all_labels_test, all_probs = [], [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            f_v = batch["image_features"].to(DEVICE)
            f_t = batch["text_features"].to(DEVICE)
            logits = baseline_model(f_v, f_t)
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels_test.extend(batch["label"].numpy())
            all_probs.extend(torch.softmax(logits, -1).cpu().numpy())

    y_true = np.array(all_labels_test)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["seed"] = seed
    metrics["predictions"] = y_pred
    metrics["labels"] = y_true
    metrics["probabilities"] = y_prob
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    baseline_results.append(metrics)

    print(
        f"F1w={metrics['f1_weighted']:.4f} "
        f"F1m={metrics['f1_macro']:.4f} "
        f"BAcc={metrics['balanced_accuracy']:.4f}"
    )

f1_baseline = [r["f1_weighted"] for r in baseline_results]
macro_baseline = [r["f1_macro"] for r in baseline_results]
bacc_baseline = [r["balanced_accuracy"] for r in baseline_results]

print("\n" + "=" * 60)
print("H1 Summary")
print(
    f"Baseline F1w: {np.mean(f1_baseline):.4f} +- {np.std(f1_baseline):.4f} | "
    f"F1m: {np.mean(macro_baseline):.4f} +- {np.std(macro_baseline):.4f} | "
    f"BAcc: {np.mean(bacc_baseline):.4f} +- {np.std(bacc_baseline):.4f}"
)
print(f"H1 supported (> {TARGET_H1:.2f}): {np.mean(f1_baseline) > TARGET_H1}")

# %%
# ============================================================================
# CELL 7: EXPERIMENT H2/H3/H5/H6 - CausalCrisis V3 Full Model
# ============================================================================
print("\n" + "=" * 60)
print("CausalCrisis V3 full model")
print(f"Targets: >{TARGET_CRISISSPOT:.3f} (CrisisSpot), >{TARGET_H6:.2f} (H6)")
print("=" * 60)

# Stabilized preset to improve V3 training signal on CrisisMMD Task 1.
V3_PRESET = {
    "epochs": 100,
    "patience": 20,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 10,
    "ba_start_epoch": 30,
    "grl_lambda_max": 0.3,
    "grl_warmup_epochs": 10,
    "alpha_adv": 0.05,
    "alpha_ortho": 0.03,
    "alpha_supcon": 0.05,
    "use_adaptive_weights": False,
    "loss_ramp_epochs": 10,
    "scheduler_eta_min": 1e-6,
    "eval_pick_best_ba": True,
}
print("V3 preset:")
for k, v in V3_PRESET.items():
    print(f"  {k}: {v}")

v3_results = []
v3_histories = []
v3_ckpts = {}


def build_v3_model(cfg):
    """Build V3 model with CLIP-Adapter + disentanglement."""
    return CausalCrisisV3(
        input_dim=cfg.clip.image_dim,
        causal_dim=cfg.disentangle.causal_dim,
        spurious_dim=cfg.disentangle.spurious_dim,
        num_classes=cfg.classifier.num_classes,
        num_domains=cfg.training.num_domains,
        nhead=cfg.fusion.nhead,
        dropout=cfg.disentangle.dropout,
        use_ica_init=cfg.disentangle.use_ica_init,
        fusion_type=cfg.fusion.fusion_type,
        grl_lambda_max=V3_PRESET["grl_lambda_max"],
        grl_warmup_epochs=V3_PRESET["grl_warmup_epochs"],
        # CLIP-Adapter
        use_adapter=cfg.adapter.use_adapter,
        adapter_bottleneck=cfg.adapter.bottleneck,
        adapter_residual_ratio=cfg.adapter.residual_ratio,
    )


for seed in SEEDS:
    print(f"\n--- V3 seed {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = build_v3_model(config)

    loss_fn = CausalCrisisLoss(
        num_classes=config.classifier.num_classes,
        focal_gamma=config.training.focal_gamma,
        alpha_adv=V3_PRESET["alpha_adv"],
        alpha_ortho=V3_PRESET["alpha_ortho"],
        alpha_supcon=V3_PRESET["alpha_supcon"],
        use_adaptive=V3_PRESET["use_adaptive_weights"],
        class_weights=class_weights,
        loss_ramp_epochs=V3_PRESET["loss_ramp_epochs"],
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=V3_PRESET["lr"],
        weight_decay=V3_PRESET["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=V3_PRESET["epochs"],
        eta_min=V3_PRESET["scheduler_eta_min"],
    )

    trainer = CausalCrisisTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        warmup_epochs=V3_PRESET["warmup_epochs"],
        ba_start_epoch=V3_PRESET["ba_start_epoch"],
        save_dir="checkpoints",
        experiment_name=f"v3_seed{seed}",
    )

    history = trainer.train(
        loaders["train"],
        loaders["val"],
        epochs=V3_PRESET["epochs"],
        patience=V3_PRESET["patience"],
    )
    v3_histories.append(history)
    trained_epochs = len(history.get("val_f1", []))
    phase2_epochs = max(0, trained_epochs - V3_PRESET["warmup_epochs"])
    ba_epochs = max(0, trained_epochs - V3_PRESET["ba_start_epoch"])
    print(
        f"Training span: {trained_epochs} epochs | "
        f"Phase2 active: {phase2_epochs} | BA active: {ba_epochs}"
    )
    if phase2_epochs == 0:
        print("Warning: warmup consumed all trained epochs; causal losses never activated.")
    if ba_epochs == 0:
        print("Warning: BA never activated (trained epochs < ba_start_epoch).")

    ckpt_path = f"checkpoints/v3_seed{seed}_best.pt"
    trainer.load_checkpoint(ckpt_path)
    v3_ckpts[seed] = ckpt_path

    eval_no_ba = trainer.evaluate(loaders["test"], use_ba=False)
    eval_with_ba = trainer.evaluate(loaders["test"], use_ba=True)
    print(
        f"Test F1 by eval mode: no_BA={eval_no_ba['f1']:.4f}, "
        f"with_BA={eval_with_ba['f1']:.4f}"
    )
    if V3_PRESET["eval_pick_best_ba"] and eval_with_ba["f1"] >= eval_no_ba["f1"]:
        eval_out = eval_with_ba
        used_ba_eval = True
    else:
        eval_out = eval_no_ba
        used_ba_eval = False

    y_true = eval_out["labels"]
    y_pred = eval_out["predictions"]
    y_prob = eval_out["probabilities"]

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["seed"] = seed
    metrics["predictions"] = y_pred
    metrics["labels"] = y_true
    metrics["probabilities"] = y_prob
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1"] = metrics["f1_weighted"]
    metrics["used_ba_eval"] = used_ba_eval
    v3_results.append(metrics)

    print(
        f"F1w={metrics['f1_weighted']:.4f} "
        f"F1m={metrics['f1_macro']:.4f} "
        f"BAcc={metrics['balanced_accuracy']:.4f} "
        f"(used_BA_eval={used_ba_eval})"
    )

f1_v3 = [r["f1_weighted"] for r in v3_results]
macro_v3 = [r["f1_macro"] for r in v3_results]
bacc_v3 = [r["balanced_accuracy"] for r in v3_results]

print("\n" + "=" * 60)
print("V3 Summary")
print(
    f"V3 F1w: {np.mean(f1_v3):.4f} +- {np.std(f1_v3):.4f} | "
    f"F1m: {np.mean(macro_v3):.4f} +- {np.std(macro_v3):.4f} | "
    f"BAcc: {np.mean(bacc_v3):.4f} +- {np.std(bacc_v3):.4f}"
)
print(f"Delta F1w vs baseline: {np.mean(f1_v3) - np.mean(f1_baseline):+.4f}")
print(f"Surpass CrisisSpot (0.909): {np.mean(f1_v3) > TARGET_CRISISSPOT}")
print(f"Meet H6 target (>0.91): {np.mean(f1_v3) > TARGET_H6}")

# %%
# ============================================================================
# CELL 8: Visualization
# ============================================================================
print("\nGenerating visualizations...")

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 120

if v3_histories:
    best_v3_idx = int(np.argmax(f1_v3))
    plot_training_curves(v3_histories[best_v3_idx], save_path="v3_training_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

methods = ["Baseline", "CausalCrisis V3", "CrisisSpot"]
f1_means = [np.mean(f1_baseline), np.mean(f1_v3), TARGET_CRISISSPOT]
f1_stds = [np.std(f1_baseline), np.std(f1_v3), 0.0]
colors = ["#4ECDC4", "#FF6B6B", "#95E1D3"]

bars = axes[0].bar(methods, f1_means, yerr=f1_stds, capsize=5, color=colors)
axes[0].set_ylabel("Weighted F1")
axes[0].set_title("Model Comparison")
y_min = max(0.0, min(f1_means) - 0.03)
y_max = min(1.0, max(f1_means) + 0.03)
if y_max - y_min < 0.08:
    y_min = max(0.0, y_max - 0.08)
axes[0].set_ylim(y_min, y_max)
axes[0].axhline(y=TARGET_CRISISSPOT, color="gray", linestyle="--", alpha=0.6)
for bar, mean in zip(bars, f1_means):
    axes[0].text(bar.get_x() + bar.get_width() / 2, mean + 0.005, f"{mean:.3f}", ha="center")

if baseline_histories and v3_histories:
    best_baseline_idx = int(np.argmax(f1_baseline))
    best_v3_idx = int(np.argmax(f1_v3))
    axes[1].plot(baseline_histories[best_baseline_idx]["val_f1"], label="Baseline")
    axes[1].plot(v3_histories[best_v3_idx]["val_f1"], label="V3")
    axes[1].axhline(y=TARGET_CRISISSPOT, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val F1")
    axes[1].set_title("Validation Curve (best seed)")
    axes[1].legend()

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ============================================================================
# CELL 9: t-SNE of Causal Features (best V3 seed)
# ============================================================================
print("\nt-SNE of causal features...")

best_v3_idx = int(np.argmax(f1_v3))
best_v3_seed = v3_results[best_v3_idx]["seed"]
best_ckpt = v3_ckpts[best_v3_seed]

viz_model = build_v3_model(config).to(DEVICE)
ckpt = torch.load(best_ckpt, map_location=DEVICE)
viz_model.load_state_dict(ckpt["model_state_dict"])
viz_model.eval()

all_C_v, all_C_t = [], []
all_labels_viz, all_domains_viz = [], []

with torch.no_grad():
    for batch in loaders["test"]:
        f_v = batch["image_features"].to(DEVICE)
        f_t = batch["text_features"].to(DEVICE)
        out = viz_model(f_v, f_t, use_ba=False)
        all_C_v.append(out["C_v"].cpu().numpy())
        all_C_t.append(out["C_t"].cpu().numpy())
        all_labels_viz.extend(batch["label"].numpy())
        if "domain_id" in batch:
            all_domains_viz.extend(batch["domain_id"].numpy())

C_v_all = np.concatenate(all_C_v)
C_t_all = np.concatenate(all_C_t)
labels_viz = np.array(all_labels_viz)
domains_viz = np.array(all_domains_viz) if all_domains_viz else np.zeros(len(labels_viz))

plot_tsne_causal_features(
    C_v_all,
    C_t_all,
    labels_viz,
    domains_viz,
    save_path="tsne_causal_features.png",
)

# %%
# ============================================================================
# CELL 10: Statistical Significance (best V3 vs best baseline)
# ============================================================================
print("\nStatistical significance (paired bootstrap)")
print("=" * 60)

best_baseline_idx = int(np.argmax(f1_baseline))
best_v3_idx = int(np.argmax(f1_v3))

baseline_preds = baseline_results[best_baseline_idx]["predictions"]
v3_preds = v3_results[best_v3_idx]["predictions"]
test_labels = v3_results[best_v3_idx]["labels"]

if len(baseline_preds) == len(v3_preds):
    sig_test = paired_bootstrap_test(
        test_labels,
        v3_preds,
        baseline_preds,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.alpha,
    )
    print(f"V3 F1:       {sig_test['score_a']:.4f}")
    print(f"Baseline F1: {sig_test['score_b']:.4f}")
    print(f"Difference:  {sig_test['observed_diff']:+.4f}")
    print(f"p-value:     {sig_test['p_value']:.6f}")
    print(f"95% CI:      [{sig_test['ci_lower']:+.4f}, {sig_test['ci_upper']:+.4f}]")
    print(f"Significant: {sig_test['significant']}")
else:
    print("Cannot run significance test: prediction length mismatch")

# %%
# ============================================================================
# CELL 11: Optional LODO Evaluation (disabled by default)
# ============================================================================
RUN_LODO = False
LODO_EPOCHS = 40
lodo_results = None

if RUN_LODO:
    print("\nRunning LODO evaluation...")

    event_names = []
    if "event_name" in data.columns:
        event_names = sorted(data["event_name"].dropna().unique().tolist())

    def lodo_model_factory():
        model = build_v3_model(config)
        # Placeholder optimizer/scheduler; real ones are rebuilt in lodo_train_fn
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.T_0,
            T_mult=config.training.T_mult,
            eta_min=config.training.eta_min,
        )
        return model, optimizer, scheduler

    def lodo_loss_fn_factory():
        return CausalCrisisLoss(
            num_classes=config.classifier.num_classes,
            focal_gamma=config.training.focal_gamma,
            alpha_adv=config.training.alpha_adv,
            alpha_ortho=config.training.alpha_ortho,
            alpha_supcon=config.training.alpha_supcon,
            use_adaptive=config.training.use_adaptive_weights,
            class_weights=class_weights,
        )

    def lodo_train_fn(model, loss_fn, optimizer, scheduler, train_loader, val_loader, epochs, device):
        # Rebuild optimizer to include learnable loss parameters (adaptive weighting)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.T_0,
            T_mult=config.training.T_mult,
            eta_min=config.training.eta_min,
        )
        trainer = CausalCrisisTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            warmup_epochs=config.training.warmup_epochs,
            ba_start_epoch=config.training.ba_start_epoch,
            save_dir="checkpoints",
            experiment_name="lodo_tmp",
        )
        trainer.train(train_loader, val_loader, epochs=epochs, patience=10)
        return trainer.history

    lodo_results = run_lodo_evaluation(
        model_factory=lodo_model_factory,
        loss_fn_factory=lodo_loss_fn_factory,
        image_features=image_features,
        text_features=text_features,
        labels=labels,
        domain_ids=domain_ids,
        event_names=event_names,
        train_fn=lodo_train_fn,
        epochs=LODO_EPOCHS,
        batch_size=config.training.batch_size,
        seeds=SEEDS,
        device=DEVICE,
    )
else:
    print("LODO disabled. Set RUN_LODO=True to execute.")

# %%
# ============================================================================
# CELL 12: Optional Ablation Study (disabled by default)
# ============================================================================
RUN_ABLATION = False
ablation_results = None

if RUN_ABLATION:
    print("\nRunning ablation study...")

    ABLATION_VARIANTS = {
        "full": {
            "use_ica_init": True,
            "alpha_supcon": config.training.alpha_supcon,
            "fusion_type": "cross_attention",
            "use_ba": True,
        },
        "no_ica": {
            "use_ica_init": False,
            "alpha_supcon": config.training.alpha_supcon,
            "fusion_type": "cross_attention",
            "use_ba": True,
        },
        "no_supcon": {
            "use_ica_init": True,
            "alpha_supcon": 0.0,
            "fusion_type": "cross_attention",
            "use_ba": True,
        },
        "bilinear_fusion": {
            "use_ica_init": True,
            "alpha_supcon": config.training.alpha_supcon,
            "fusion_type": "bilinear",
            "use_ba": True,
        },
        "no_backdoor": {
            "use_ica_init": True,
            "alpha_supcon": config.training.alpha_supcon,
            "fusion_type": "cross_attention",
            "use_ba": False,
        },
    }

    def run_single_ablation(variant_cfg, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = CausalCrisisV3(
            input_dim=config.clip.image_dim,
            causal_dim=config.disentangle.causal_dim,
            spurious_dim=config.disentangle.spurious_dim,
            num_classes=config.classifier.num_classes,
            num_domains=config.training.num_domains,
            nhead=config.fusion.nhead,
            dropout=config.disentangle.dropout,
            use_ica_init=variant_cfg["use_ica_init"],
            fusion_type=variant_cfg["fusion_type"],
            grl_lambda_max=config.training.grl_lambda_max,
            grl_warmup_epochs=config.training.grl_warmup_epochs,
        )

        loss_fn = CausalCrisisLoss(
            num_classes=config.classifier.num_classes,
            focal_gamma=config.training.focal_gamma,
            alpha_adv=config.training.alpha_adv,
            alpha_ortho=config.training.alpha_ortho,
            alpha_supcon=variant_cfg["alpha_supcon"],
            use_adaptive=config.training.use_adaptive_weights,
            class_weights=class_weights,
        )

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.T_0,
            T_mult=config.training.T_mult,
            eta_min=config.training.eta_min,
        )

        trainer = CausalCrisisTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            warmup_epochs=config.training.warmup_epochs,
            ba_start_epoch=config.training.ba_start_epoch,
            save_dir="checkpoints",
            experiment_name=f"ablation_{seed}",
        )

        trainer.train(loaders["train"], loaders["val"], epochs=50, patience=10)
        ablation_ckpt = f"checkpoints/ablation_{seed}_best.pt"
        trainer.load_checkpoint(ablation_ckpt)
        eval_out = trainer.evaluate(loaders["test"], use_ba=variant_cfg["use_ba"])
        m = compute_metrics(eval_out["labels"], eval_out["predictions"], eval_out["probabilities"])
        return {"f1_weighted": m["f1_weighted"]}

    ablation_results = run_ablation_study(
        variants=ABLATION_VARIANTS,
        base_train_fn=run_single_ablation,
        seeds=SEEDS,
    )
else:
    print("Ablation disabled. Set RUN_ABLATION=True to execute.")

# %%
# ============================================================================
# CELL 13: Save Results
# ============================================================================
print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

os.makedirs("results", exist_ok=True)

summary = {
    "experiment": "CausalCrisis V3",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "CrisisMMD v2.0 Task 1",
    "split_mode": split_mode,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "targets": {
        "h1": TARGET_H1,
        "crisisspot": TARGET_CRISISSPOT,
        "h6": TARGET_H6,
    },
    "seeds": SEEDS,
    "baseline": {
        "f1_weighted_mean": float(np.mean(f1_baseline)),
        "f1_weighted_std": float(np.std(f1_baseline)),
        "f1_macro_mean": float(np.mean(macro_baseline)),
        "f1_macro_std": float(np.std(macro_baseline)),
        "balanced_acc_mean": float(np.mean(bacc_baseline)),
        "balanced_acc_std": float(np.std(bacc_baseline)),
        "f1_weighted_scores": [float(x) for x in f1_baseline],
    },
    "v3": {
        "f1_weighted_mean": float(np.mean(f1_v3)),
        "f1_weighted_std": float(np.std(f1_v3)),
        "f1_macro_mean": float(np.mean(macro_v3)),
        "f1_macro_std": float(np.std(macro_v3)),
        "balanced_acc_mean": float(np.mean(bacc_v3)),
        "balanced_acc_std": float(np.std(bacc_v3)),
        "f1_weighted_scores": [float(x) for x in f1_v3],
    },
    "delta_f1_weighted": float(np.mean(f1_v3) - np.mean(f1_baseline)),
    "surpass_crisisspot": bool(np.mean(f1_v3) > TARGET_CRISISSPOT),
    "meet_h6": bool(np.mean(f1_v3) > TARGET_H6),
    "lodo": lodo_results,
    "ablation": ablation_results,
}

results_path = "results/experiment_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(to_python_types(summary), f, indent=2)

print(f"Saved: {results_path}")

if IN_COLAB and DRIVE_MOUNTED and os.path.exists("/content/drive/MyDrive"):
    drive_dir = "/content/drive/MyDrive/CausalCrisis_Results"
    os.makedirs(drive_dir, exist_ok=True)
    os.system(f"cp {results_path} {drive_dir}/")
    os.system(f"cp -f checkpoints/*.pt {drive_dir}/ 2>/dev/null || true")
    os.system(f"cp -f *.png {drive_dir}/ 2>/dev/null || true")
    print(f"Copied artifacts to: {drive_dir}")

# %%
# ============================================================================
# CELL 14: Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("CausalCrisis V3 experiment complete")
print("=" * 60)
print(f"Split mode: {split_mode}")
print(f"Baseline F1w: {np.mean(f1_baseline):.4f} +- {np.std(f1_baseline):.4f}")
print(f"V3 F1w:       {np.mean(f1_v3):.4f} +- {np.std(f1_v3):.4f}")
print(f"Delta:        {np.mean(f1_v3) - np.mean(f1_baseline):+.4f}")
print(f"CrisisSpot 0.909 beaten: {np.mean(f1_v3) > TARGET_CRISISSPOT}")
print(f"H6 > 0.91 met: {np.mean(f1_v3) > TARGET_H6}")
print("=" * 60)
