# Causal Crisis Generalization for Multimodal Disaster Posts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository implements the **Causal Multimodal Reasoning Framework for Crisis Generalization**, featuring a novel integration of Graph Neural Networks (GNNs), Causal Intervention mechanisms (do-calculus), and Multimodal Disentanglement. 

The framework focuses on **Out-of-Distribution (OOD)** generalizability to unseen disasters by separating invariant, event-agnostic _causal features_ from heavily biased _spurious features_.

## 📖 Table of Contents
1. [Repository Structure](#repository-structure)
2. [Prerequisites & Dataset Setup](#prerequisites--dataset-setup)
3. [Running on Google Colab (Recommended)](#running-on-google-colab-recommended)
4. [Training the Model](#training-the-model)
5. [Evaluation protocols](#evaluation-protocols)

---

## 📁 Repository Structure

We follow a rigorous scientific machine learning repository standard to ensure clear separation of concerns, reproducibility, and minimal friction during experimentation.

```plaintext
CrisisSummarization/
│
├── data/                   # (IGNORED IN GIT) Ensure your large data lives here
│   ├── raw/                # Original datasets (e.g. CrisisMMD_v2.0 root folder)
│   ├── processed/          # Cached CLIP features (.npy), Faiss indices, etc.
│   └── splits/             # Train/Val/Test metadata (e.g. crisis_mmd_splits.tsv)
│
├── checkpoints/            # (IGNORED IN GIT) Saved PyTorch models (*.pt)
├── results/                # (IGNORED IN GIT) Raw outputs like CSV logs, metrics
├── notebooks/              # Jupyter/Colab notebooks for exploring & iterating
│
├── src/                    # Core mathematical components and algorithms
│   ├── models/             # PyTorch model definitions (CausalCrisisModel, Loss, Disentangler)
│   ├── trainers/           # Complex training loops (CausalCrisisTrainer, Ablation & LODO loops)
│   └── utils/              # Plotting, statistical tests, metrics logic
│
├── docs/                   # Knowledge architecture, design, implementations, papers
├── evaluation/             # Independent scripts dedicated strictly to plotting/reporting results
│
├── requirements.txt        # Python dependency declarations
└── README.md               # You are here!
```

---

## 🛠️ Prerequisites & Dataset Setup

This project fundamentally requires the **CrisisMMD_v2.0** dataset for multi-modal text-image input.

1. **Clone the project:**
   ```bash
   git clone https://github.com/YourUsername/CrisisSummarization.git
   cd CrisisSummarization
   ```

2. **Acquire CrisisMMD_v2.0**:
   Download the original CrisisMMD dataset. Move the data into `data/raw/CrisisMMD_v2.0`.
    Ensure the annotation TSV files (like `crisis_mmd_splits.tsv`) specify columns at least for `image_path`, `tweet_text`, `label`, and `event_name` (crucial for Causal LODO splits).

---

## ☁️ Running on Google Colab (Recommended)

Given the heavy computational graph of PyTorch Sparse Multi-Modal operations, **Google Colab (A100 or T4 GPU)** is highly recommended.

**Step 1:** Upload your raw Dataset to Google Drive (e.g., `MyDrive/datasets/CrisisMMD_v2.0`).

**Step 2:** Open a standard Colab Notebook, mount your drive, and seamlessly run our framework straight from Github by pasting the following into a Colab cell:

```python
import os
import sys

# 1. Mount Google Drive containing datasets
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone this systematic repository
!git clone https://github.com/YourUsername/CrisisSummarization.git
%cd CrisisSummarization

# 3. Install requirements
!pip install -r requirements.txt

# 4. Inject Repo Path for inner absolute importing
REPO_DIR = "/content/CrisisSummarization"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
```

---

## 🚀 Training the Model

The framework abstracts away the complexity of training. You can invoke comprehensive testing directly via `src.trainers.causal_crisis_trainer`:

```python
from src.trainers.causal_crisis_trainer import run_ablation_suite, run_lodo_all_experiments

# 1. Standard Ablation with 50-shot setting (Full, No Attn, No Causal, etc.)
run_ablation_suite(
    dataset_path='/content/drive/MyDrive/datasets/CrisisMMD_v2.0',
    seeds=[42, 123, 456],
    tasks=["task1"], 
    few_shot_sizes=[50],
    device="cuda",
    results_csv="./results/ablation_test_results.csv"  # Outputs are saved locally out of git's sight
)

# 2. Leave-One-Disaster-Out (LODO) Experiment for pure Out-of-Distribution 
run_lodo_all_experiments(
    dataset_path='/content/drive/MyDrive/datasets/CrisisMMD_v2.0',
    seeds=[42], 
    task="task1", 
    size=500,
    device="cuda",
    results_csv="./results/lodo_test_results.csv"
)
```

---

## 🧪 Evaluation Protocols
After running the trainer, the outputs will be safely flushed into `/results`. You can visualize the Causal Feature disentanglement or statistically test metrics by navigating to the `evaluation/` directory routines (e.g., `metrics.py`), where we compute the `ttest_rel` for determining the statistical significance of using Intervention.
