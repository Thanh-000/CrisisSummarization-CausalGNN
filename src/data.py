# ============================================================================
# CausalCrisis V3 — Data Pipeline
# CrisisMMD v2.0 loading, CLIP feature caching, stratified splits
# ============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ============================================================================
# CrisisMMD Dataset Reader
# ============================================================================
def load_crisismmd_annotations(
    data_dir: str,
    task: str = "task1"
) -> pd.DataFrame:
    """
    Load CrisisMMD v2.0 annotations.
    
    Args:
        data_dir: đường dẫn tới thư mục CrisisMMD
        task: "task1" (informative), "task2" (humanitarian), "task3" (damage)
    
    Returns:
        DataFrame with columns: image_path, text, label, label_name, event_name
    """
    task_file_map = {
        "task1": "crisismmd_datasplit_all/task_informative_text_img_agreed_lab",
        "task2": "crisismmd_datasplit_all/task_humanitarian_text_img_agreed_lab",
        "task3": "crisismmd_datasplit_all/task_damage_text_img_agreed_lab",
    }
    task_keywords = {
        "task1": ["informative", "text", "img"],
        "task2": ["humanitarian", "text", "img"],
        "task3": ["damage", "text", "img"],
    }

    def detect_sep(filepath: str) -> str:
        """Best-effort delimiter detection for annotation files."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
        except Exception:
            return "\t"
        tab_count = first_line.count("\t")
        comma_count = first_line.count(",")
        return "\t" if tab_count >= comma_count else ","

    def find_annotation_file(split: str) -> Optional[str]:
        """Find best annotation file for (task, split) if canonical path is missing."""
        canonical = os.path.join(data_dir, f"{task_file_map[task]}_{split}.tsv")
        if os.path.exists(canonical):
            return canonical

        split_aliases = {
            "train": ["train"],
            "dev": ["dev", "val", "valid", "validation"],
            "test": ["test"],
        }[split]

        candidates = []
        for dirpath, _, filenames in os.walk(data_dir):
            for fn in filenames:
                fn_lower = fn.lower()
                if not fn_lower.endswith((".tsv", ".csv", ".txt")):
                    continue
                if not any(tag in fn_lower for tag in split_aliases):
                    continue
                score = 0
                for kw in task_keywords.get(task, []):
                    if kw in fn_lower:
                        score += 2
                if "task1" in fn_lower and task == "task1":
                    score += 3
                if "task2" in fn_lower and task == "task2":
                    score += 3
                if "task3" in fn_lower and task == "task3":
                    score += 3
                candidates.append((score, os.path.join(dirpath, fn)))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_path = candidates[0][1]
        print(f"ℹ️ Using discovered {task} {split} file: {best_path}")
        return best_path
    
    # Label mapping cho từng task
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
    
    # Đọc từng split
    records = []
    for split in ["train", "dev", "test"]:
        filepath = find_annotation_file(split)
        if not filepath or not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            continue
        
        sep = detect_sep(filepath)
        df = pd.read_csv(filepath, sep=sep)

        cols = list(df.columns)

        def choose_best_column(kind: str) -> Optional[str]:
            best_col = None
            best_score = -10**9
            for col in cols:
                lc = str(col).lower().strip()
                score = 0
                if kind == "image":
                    if "image_path" in lc:
                        score += 120
                    if lc in {"image", "img"}:
                        score += 100
                    if "image" in lc or "img" in lc:
                        score += 50
                    if "id" in lc:
                        score -= 80
                elif kind == "text":
                    if "tweet_text" in lc:
                        score += 120
                    if lc == "text":
                        score += 100
                    if "text" in lc:
                        score += 50
                elif kind == "label":
                    if lc in {"label_name", "label"}:
                        score += 120
                    if "label" in lc:
                        score += 60
                    if "id" in lc:
                        score -= 60
                elif kind == "event":
                    if lc == "event_name":
                        score += 120
                    if "event" in lc:
                        score += 50
                    if "id" in lc:
                        score -= 40

                if score > best_score:
                    best_score = score
                    best_col = col

            return best_col if best_score > 0 else None

        image_col = choose_best_column("image")
        text_col = choose_best_column("text")
        label_col = choose_best_column("label")
        event_col = choose_best_column("event")

        # Build a canonical dataframe to avoid duplicate-column rename issues.
        norm = pd.DataFrame(index=df.index)
        if image_col is not None:
            norm["image_path"] = df[image_col]
        if text_col is not None:
            norm["text"] = df[text_col]
        else:
            norm["text"] = ""
        if label_col is not None:
            norm["label_name"] = df[label_col]
        if event_col is not None:
            norm["event_name"] = df[event_col]
        norm["split"] = split

        if image_col is None or label_col is None:
            print(
                f"⚠️ Skip split {split}: cannot identify required columns "
                f"(image_col={image_col}, label_col={label_col}) from {filepath}"
            )
            continue
        
        # Map labels
        if "label_name" in norm.columns:
            label_map = label_maps.get(task, {})
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
        
        # Fix image paths
        if "image_path" in norm.columns:
            def to_abs_image_path(x):
                if pd.isna(x):
                    return ""
                p = str(x).strip()
                if not p:
                    return ""
                if os.path.isabs(p):
                    return p
                return os.path.join(data_dir, p.lstrip("./\\"))
            norm["image_path"] = norm["image_path"].apply(to_abs_image_path)
        
        records.append(norm)
    
    if not records:
        raise FileNotFoundError(f"No CrisisMMD files found in {data_dir}")
    
    data = pd.concat(records, ignore_index=True)
    
    # Thêm event_id (domain index)
    if "event_name" in data.columns:
        events = sorted(data["event_name"].unique())
        event_to_id = {e: i for i, e in enumerate(events)}
        data["domain_id"] = data["event_name"].map(event_to_id)
    
    print(f"✅ Loaded CrisisMMD {task}: {len(data)} samples, "
          f"{data['label'].nunique()} classes")
    print(f"   Class distribution: {dict(Counter(data['label'].values))}")
    
    if "event_name" in data.columns:
        print(f"   Events ({len(events)}): {events}")
    
    return data


# ============================================================================
# CLIP Feature Extraction & Caching
# ============================================================================
def extract_and_cache_clip_features(
    data: pd.DataFrame,
    cache_dir: str = "cached_features",
    model_name: str = "ViT-L/14",
    batch_size: int = 64,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLIP features và cache vào .npy files.
    
    Returns:
        image_features: (N, 768)
        text_features: (N, 768)
    """
    cache_path_img = os.path.join(cache_dir, f"clip_{model_name.replace('/', '_')}_image.npy")
    cache_path_txt = os.path.join(cache_dir, f"clip_{model_name.replace('/', '_')}_text.npy")
    cache_path_idx = os.path.join(cache_dir, "indices.json")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Kiểm tra cache có sẵn không
    if os.path.exists(cache_path_img) and os.path.exists(cache_path_txt):
        print(f"📦 Loading cached features from {cache_dir}")
        image_features = np.load(cache_path_img)
        text_features = np.load(cache_path_txt)
        print(f"   Image features: {image_features.shape}")
        print(f"   Text features: {text_features.shape}")
        return image_features, text_features
    
    print(f"🔄 Extracting CLIP features ({model_name})...")
    
    try:
        import open_clip
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Cần install: pip install open_clip_torch pillow"
        )
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name.replace("/", "-"),
        pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer(model_name.replace("/", "-"))
    model = model.to(device).eval()
    
    all_image_features = []
    all_text_features = []
    
    # Batch processing
    n_batches = (len(data) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data))
        batch = data.iloc[start:end]
        
        # Image features
        images = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                # Nếu ảnh bị lỗi, dùng dummy
                images.append(torch.zeros(3, 224, 224))
        
        image_batch = torch.stack(images).to(device)
        
        # Text features
        texts = batch["text"].fillna("").tolist()
        text_tokens = tokenizer(texts).to(device)
        
        with torch.no_grad():
            img_feat = model.encode_image(image_batch)
            txt_feat = model.encode_text(text_tokens)
            
            # L2 normalize
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        
        all_image_features.append(img_feat.cpu().numpy())
        all_text_features.append(txt_feat.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   Batch {batch_idx + 1}/{n_batches}")
    
    image_features = np.concatenate(all_image_features)
    text_features = np.concatenate(all_text_features)
    
    # Cache
    np.save(cache_path_img, image_features)
    np.save(cache_path_txt, text_features)
    
    # Save index mapping
    with open(cache_path_idx, "w") as f:
        json.dump({"model": model_name, "n_samples": len(data)}, f)
    
    print(f"✅ Features cached: image {image_features.shape}, text {text_features.shape}")
    
    # Giải phóng GPU memory
    del model
    torch.cuda.empty_cache()
    
    return image_features, text_features


# ============================================================================
# PyTorch Dataset
# ============================================================================
class CrisisMMDDataset(Dataset):
    """
    PyTorch Dataset cho CrisisMMD với pre-cached CLIP features.
    """
    
    def __init__(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        labels: np.ndarray,
        domain_ids: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            self.image_features = torch.FloatTensor(image_features[indices])
            self.text_features = torch.FloatTensor(text_features[indices])
            self.labels = torch.LongTensor(labels[indices])
            if domain_ids is not None:
                self.domain_ids = torch.LongTensor(domain_ids[indices])
            else:
                self.domain_ids = None
        else:
            self.image_features = torch.FloatTensor(image_features)
            self.text_features = torch.FloatTensor(text_features)
            self.labels = torch.LongTensor(labels)
            if domain_ids is not None:
                self.domain_ids = torch.LongTensor(domain_ids)
            else:
                self.domain_ids = None
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "image_features": self.image_features[idx],
            "text_features": self.text_features[idx],
            "label": self.labels[idx],
        }
        if self.domain_ids is not None:
            item["domain_id"] = self.domain_ids[idx]
        return item


# ============================================================================
# Stratified Splits
# ============================================================================
def create_stratified_splits(
    labels: np.ndarray,
    domain_ids: Optional[np.ndarray] = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tạo stratified train/val/test splits.
    Stratify theo cả label và domain nếu có.
    """
    from sklearn.model_selection import train_test_split
    
    n = len(labels)
    indices = np.arange(n)
    
    # Tạo stratification key
    if domain_ids is not None:
        strat_key = labels * 100 + domain_ids  # Unique combo
    else:
        strat_key = labels
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio,
        stratify=strat_key, random_state=seed
    )
    
    # Second split: train vs val
    val_size = val_ratio / (1 - test_ratio)
    strat_key_tv = strat_key[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size,
        stratify=strat_key_tv, random_state=seed
    )
    
    print(f"📊 Splits — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx


# ============================================================================
# LODO (Leave-One-Disaster-Out) Split
# ============================================================================
def create_lodo_splits(
    domain_ids: np.ndarray,
    target_domain: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LODO split: train trên tất cả domains trừ target_domain.
    
    Returns:
        train_indices, test_indices
    """
    all_indices = np.arange(len(domain_ids))
    test_mask = domain_ids == target_domain
    
    train_indices = all_indices[~test_mask]
    test_indices = all_indices[test_mask]
    
    print(f"🌍 LODO — Target domain {target_domain}: "
          f"Train {len(train_indices)}, Test {len(test_indices)}")
    
    return train_indices, test_indices


# ============================================================================
# DataLoader Factory
# ============================================================================
def create_dataloaders(
    image_features: np.ndarray,
    text_features: np.ndarray,
    labels: np.ndarray,
    domain_ids: Optional[np.ndarray] = None,
    train_idx: Optional[np.ndarray] = None,
    val_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create dataloaders for train/val/test."""
    
    loaders = {}
    
    if train_idx is not None:
        train_ds = CrisisMMDDataset(
            image_features, text_features, labels, domain_ids, train_idx
        )
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
    
    if val_idx is not None:
        val_ds = CrisisMMDDataset(
            image_features, text_features, labels, domain_ids, val_idx
        )
        loaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    if test_idx is not None:
        test_ds = CrisisMMDDataset(
            image_features, text_features, labels, domain_ids, test_idx
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    return loaders


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights dựa trên frequency."""
    counts = Counter(labels)
    total = len(labels)
    # Keep weight vector aligned with class index space [0..max_label].
    # This avoids shape mismatches when some classes are absent in a split.
    n_classes = int(np.max(labels)) + 1 if len(labels) > 0 else 0
    if n_classes == 0:
        return torch.FloatTensor([])
    
    weights = []
    for c in range(n_classes):
        cls_count = counts.get(c, 0)
        if cls_count == 0:
            # Missing class in this split -> ignore in CE/Focal via zero weight.
            w = 0.0
        else:
            # Inverse frequency
            w = total / (n_classes * cls_count)
        weights.append(w)
    
    weights = torch.FloatTensor(weights)
    # Normalize only when non-zero sum exists.
    if weights.sum() > 0:
        weights = weights / weights.sum() * n_classes
    
    print(f"⚖️ Class weights: {weights.tolist()}")
    return weights


# ============================================================================
# 3-Modality Dataset (V4 — image + text + LLaVA)
# ============================================================================
class CrisisMMD3ModalDataset(Dataset):
    """Dataset supporting 3 modalities: image + text + llava (optional)."""

    def __init__(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        llava_features: Optional[np.ndarray],
        labels: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            self.image_features = torch.FloatTensor(image_features[indices])
            self.text_features = torch.FloatTensor(text_features[indices])
            self.labels = torch.LongTensor(labels[indices])
            if llava_features is not None:
                self.llava_features = torch.FloatTensor(llava_features[indices])
            else:
                self.llava_features = None
        else:
            self.image_features = torch.FloatTensor(image_features)
            self.text_features = torch.FloatTensor(text_features)
            self.labels = torch.LongTensor(labels)
            if llava_features is not None:
                self.llava_features = torch.FloatTensor(llava_features)
            else:
                self.llava_features = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "image_features": self.image_features[idx],
            "text_features": self.text_features[idx],
            "label": self.labels[idx],
        }
        if self.llava_features is not None:
            item["llava_features"] = self.llava_features[idx]
        return item


def create_3modal_loaders(
    image_feats: np.ndarray,
    text_feats: np.ndarray,
    llava_feats: Optional[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for 3-modality experiments."""
    loaders = {}

    for name, idx, shuffle in [
        ("train", train_idx, True),
        ("val", val_idx, False),
        ("test", test_idx, False),
    ]:
        ds = CrisisMMD3ModalDataset(
            image_feats, text_feats, llava_feats, labels, idx
        )
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            drop_last=(name == "train"),
        )

    return loaders

