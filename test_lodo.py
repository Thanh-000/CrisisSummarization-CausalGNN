import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_lodo_all_experiments
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = "c:/Users/Admin/CrisisMMD_v2.0/crisismmd_datasplit_all"

run_lodo_all_experiments(
    dataset_path=data_path,
    task="task1",
    size="all",  # use "all" to get higher accuracy!
    seeds=[42],
    device=device,
    results_csv="causal_results/lodo_results_fix.csv",
    variants_to_run=["GEDA_baseline", "Causal_Full"] 
)
