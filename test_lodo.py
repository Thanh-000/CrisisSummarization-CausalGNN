import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_lodo_all_experiments

parser = argparse.ArgumentParser(description="Run LODO experiments")
parser.add_argument("--task", type=str, default="task1", help="Task name (task1, task2, task3)")
parser.add_argument("--data_path", type=str, default="/content/datasets/CrisisMMD_v2.0", help="Path to dataset")
parser.add_argument("--results_csv", type=str, default="/content/lodo_results.csv", help="Output CSV path")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

variants = ["GEDA_baseline", "Causal_Full"]

run_lodo_all_experiments(
    dataset_path=args.data_path,
    task=args.task,
    size="all",
    seeds=[42],
    device=device,
    results_csv=args.results_csv,
    variants_to_run=variants
)
