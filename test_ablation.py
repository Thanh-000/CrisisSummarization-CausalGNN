import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_lodo_all_experiments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = "/content/datasets/CrisisMMD_v2.0"

variants = [
    "GEDA_baseline", 
    "Causal_NoInt",  
    "Causal_NoGraph",
    "Causal_NoAttn"
]

print(f"Starting Ablation Study (OOD/LODO) for variants: {variants}")

run_lodo_all_experiments(
    dataset_path=data_path,
    task="task1",
    size="all",
    seed=42,
    device=device,
    results_csv="/content/lodo_ablation_results.csv",
    variants_to_run=variants
)

print("Ablation study complete! Results saved to /content/lodo_ablation_results.csv")
