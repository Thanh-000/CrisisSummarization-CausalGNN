import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_causal_experiment
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_path = "/content/datasets/CrisisMMD_v2.0"

run_causal_experiment(
    dataset_path=data_path,
    task="task1",
    seed=42,
    n_labeled="all",
    device=device,
    results_csv="indist_results.csv",
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
