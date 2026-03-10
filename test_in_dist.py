import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_causal_experiment

parser = argparse.ArgumentParser(description="Run In-Distribution experiments")
parser.add_argument("--task", type=str, default="task1", help="Task name (task1, task2, task3)")
parser.add_argument("--data_path", type=str, default="/content/datasets/CrisisMMD_v2.0", help="Path to dataset")
parser.add_argument("--results_csv", type=str, default="/content/indist_results.csv", help="Output CSV path")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

run_causal_experiment(
    dataset_path=args.data_path,
    task=args.task,
    seed=42,
    n_labeled="all",
    device=device,
    results_csv=args.results_csv,
    variant_name="Causal_Full",
    use_graph=True,
    use_attention=True,
    use_mtl=False,
    use_causal=True,
    use_intervention=True,
    use_causal_graph=False,
    lodo_event=None,
    max_epochs=300,
    patience=30
)
