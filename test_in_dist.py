import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from trainers.causal_crisis_trainer import run_causal_experiment

def run_indist_study(dataset_path, task, seed, device, results_csv):
    variants = [
        {"name": "GEDA_baseline", "causal": False, "int": False, "graph": True,  "attn": True},
        {"name": "Causal_NoGraph","causal": True,  "int": True,  "graph": False, "attn": True},
    ]

    print(f"Starting In-Distribution Study for task: {task}")
    for v in variants:
        run_causal_experiment(
            dataset_path=dataset_path,
            task=task,
            seed=seed,
            n_labeled="all",
            device=device,
            results_csv=results_csv,
            variant_name=v["name"],
            use_causal=v["causal"],
            use_intervention=v["int"],
            use_causal_graph=v["causal"],
            use_graph=v["graph"],
            use_attention=v["attn"],
            lodo_event=None,
            max_epochs=200,
            patience=20
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run In-Distribution experiments")
    parser.add_argument("--task", type=str, default="task1", help="Task name (task1, task2, task3)")
    parser.add_argument("--data_path", type=str, default="/content/CrisisMMD_v2.0", help="Path to dataset")
    parser.add_argument("--results_csv", type=str, default="/content/indist_results.csv", help="Output CSV path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_indist_study(
        dataset_path=args.data_path,
        task=args.task,
        seed=42,
        device=device,
        results_csv=args.results_csv
    )
    print(f"In-Distribution study complete! Results saved to {args.results_csv}")
