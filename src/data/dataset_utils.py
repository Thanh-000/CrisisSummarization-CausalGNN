"""
dataset_utils.py — Tiện ích tải và quản lý CrisisMMD v2.0 Dataset
Dùng chung cho geda_notebook.ipynb và causal_crisis_notebook.ipynb

Cách tải giống repo GNN gốc:
  https://github.com/jdnascim/mm-class-for-disaster-data-with-gnn
"""

import os
import subprocess
import glob


# === URL chính thức từ CrisisNLP (giống repo GNN) ===
DATASET_URL = "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"


def find_dataset(drive_dir=None):
    """Tìm dataset CrisisMMD v2.0 đã có sẵn trên hệ thống.
    
    Returns:
        str or None: đường dẫn dataset nếu tìm thấy
    """
    candidates = [
        "/content/datasets/CrisisMMD_v2.0",
        "/content/CrisisMMD_v2.0",
        "/content/data/CrisisMMD_v2.0",  # giống repo GNN
    ]
    if drive_dir:
        candidates += [
            f"{drive_dir}/CrisisMMD_v2.0",
            "/content/drive/MyDrive/CrisisMMD_v2.0",
            "/content/drive/MyDrive/datasets/CrisisMMD_v2.0",
        ]
    
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def download_dataset(target_dir="/content/datasets"):
    """Tải CrisisMMD v2.0 từ URL gốc bằng wget (giống repo GNN).
    
    Args:
        target_dir: thư mục chứa dataset sau khi giải nén
        
    Returns:
        str: đường dẫn tới dataset đã tải
    """
    os.makedirs(target_dir, exist_ok=True)
    dataset_path = os.path.join(target_dir, "CrisisMMD_v2.0")
    tar_path = os.path.join(target_dir, "CrisisMMD_v2.0.tar.gz")
    
    if os.path.isdir(dataset_path):
        print(f"  ✅ Dataset đã có: {dataset_path}")
        return dataset_path
    
    # Download bằng wget (nhanh hơn Python requests trên Colab)
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1.8e9:
        print(f"  📥 Đang tải CrisisMMD v2.0 (~2GB)...")
        print(f"     URL: {DATASET_URL}")
        ret = subprocess.run(
            ["wget", "-q", "--show-progress", "-c", "-O", tar_path, DATASET_URL],
            check=False,
        )
        if ret.returncode != 0:
            print(f"  ❌ wget thất bại! Thử tải thủ công:")
            print(f"     !wget -c -O {tar_path} {DATASET_URL}")
            return None
    else:
        print(f"  ✅ Archive đã có: {tar_path} ({os.path.getsize(tar_path)/1e9:.2f} GB)")
    
    # Giải nén bằng system tar (nhanh hơn Python tarfile)
    print(f"  📦 Đang giải nén...")
    ret = subprocess.run(
        ["tar", "xzf", tar_path, "-C", target_dir],
        check=False,
    )
    if ret.returncode != 0:
        print(f"  ❌ Giải nén thất bại!")
        return None
    
    if os.path.isdir(dataset_path):
        print(f"  ✅ Dataset ready: {dataset_path}")
        return dataset_path
    else:
        print(f"  ❌ Không tìm thấy thư mục sau giải nén")
        return None


def load_from_drive(drive_dir):
    """Tìm và giải nén dataset từ Drive (zip hoặc tar.gz).
    
    Returns:
        str or None: đường dẫn dataset
    """
    archive_candidates = [
        (f"{drive_dir}/CrisisMMD_v2.0.tar.gz", "tar"),
        (f"{drive_dir}/CrisisMMD_v2.0.zip", "zip"),
        ("/content/drive/MyDrive/CrisisMMD_v2.0.tar.gz", "tar"),
        ("/content/drive/MyDrive/CrisisMMD_v2.0.zip", "zip"),
    ]
    
    for path, fmt in archive_candidates:
        if os.path.exists(path):
            print(f"  📦 Found: {path}")
            os.makedirs("/content/datasets", exist_ok=True)
            
            if fmt == "tar":
                subprocess.run(
                    ["tar", "xzf", path, "-C", "/content/datasets"],
                    check=True,
                )
            else:
                subprocess.run(
                    ["unzip", "-q", path, "-d", "/content/datasets"],
                    check=True,
                )
            
            dataset_path = "/content/datasets/CrisisMMD_v2.0"
            if os.path.isdir(dataset_path):
                print(f"  ✅ Extracted to {dataset_path}")
                return dataset_path
    
    return None


def verify_dataset(dataset_path):
    """Kiểm tra cấu trúc dataset.
    
    Returns:
        dict: thông tin dataset
    """
    info = {"path": dataset_path, "valid": False}
    
    if not os.path.isdir(dataset_path):
        print(f"  ❌ Dataset path không tồn tại: {dataset_path}")
        return info
    
    # Kiểm tra TSV files
    tsv_dir = os.path.join(dataset_path, "crisismmd_datasplit_all")
    if os.path.isdir(tsv_dir):
        tsv_files = os.listdir(tsv_dir)
        info["tsv_count"] = len(tsv_files)
        print(f"  ✅ TSV directory: {len(tsv_files)} files")
    else:
        print(f"  ⚠️ Missing: crisismmd_datasplit_all/")
    
    # Đếm images
    n_images = len(glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True))
    n_images += len(glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True))
    info["n_images"] = n_images
    print(f"  ✅ Images: {n_images:,}")
    
    # Kích thước
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(dataset_path) for f in fn
    )
    info["size_gb"] = round(total_size / 1e9, 2)
    print(f"  ✅ Size: {info['size_gb']:.2f} GB")
    
    info["valid"] = n_images > 0
    return info


def setup_dataset(source_mode="download", drive_dir=None):
    """Setup dataset đầy đủ — entry point chính.
    
    Luôn ưu tiên wget để tải dataset. Drive chỉ dùng để lưu kết quả.
    
    Args:
        source_mode: "download" hoặc "drive"
            - download: tải dataset bằng wget + lưu results trên Colab
            - drive: tải dataset bằng wget + mount Drive lưu results
        drive_dir: đường dẫn Drive (chỉ cần khi mode=drive)
        
    Returns:
        str or None: đường dẫn dataset
    """
    print("=" * 60)
    print(f"  DATASET SETUP (mode={source_mode})")
    print("=" * 60)
    
    # 1. Tìm dataset đã có (cache)
    dataset = find_dataset(drive_dir)
    if dataset:
        print(f"\n  ✅ Dataset đã có sẵn (cache): {dataset}")
        verify_dataset(dataset)
        return dataset
    
    # 2. Luôn dùng wget tải từ CrisisNLP (ưu tiên hơn Drive)
    dataset = download_dataset()
    
    # 3. Verify
    if dataset and os.path.isdir(dataset):
        print()
        verify_dataset(dataset)
        return dataset
    else:
        print(f"\n  ❌ Dataset chưa sẵn sàng!")
        print(f"     Kiểm tra kết nối mạng và thử lại")
        return None
