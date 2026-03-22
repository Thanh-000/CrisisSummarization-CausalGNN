# ============================================================================
# CausalCrisis V3 — Trainer Module
# 2-Phase training, early stopping, logging
# ============================================================================

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stopping monitor."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CausalCrisisTrainer:
    """
    Trainer cho CausalCrisis V3.
    
    2-Phase Training Protocol:
      Phase 1 (warmup_epochs): Classifier warmup (freeze adversarial)
      Phase 2 (remaining): Full joint training + BA memory building
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        warmup_epochs: int = 10,
        ba_start_epoch: int = 50,
        save_dir: str = "checkpoints",
        experiment_name: str = "causalcrisis_v3",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.ba_start_epoch = ba_start_epoch
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.history: Dict = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "lr": [], "grl_lambda": [],
            "adaptive_weights": [],
        }
        self.best_val_f1 = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for 1 epoch."""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        loss_components = {}
        
        # Update GRL lambda
        if hasattr(self.model, 'update_grl'):
            self.model.update_grl(epoch)
        
        for batch in train_loader:
            f_v = batch["image_features"].to(self.device)
            f_t = batch["text_features"].to(self.device)
            labels = batch["label"].to(self.device)
            domain_ids = batch.get("domain_id")
            if domain_ids is not None:
                domain_ids = domain_ids.to(self.device)
            
            # Phase 1: warmup (no adversarial)
            if epoch < self.warmup_epochs:
                domain_ids_input = None
            else:
                domain_ids_input = domain_ids
            
            # Forward
            output = self.model(
                f_v, f_t,
                domain_labels=domain_ids_input,
                use_ba=False,
            )
            
            # Loss (with gradual ramp-up awareness)
            losses = self.loss_fn(
                output, labels, domain_ids_input, epoch,
                warmup_epochs=self.warmup_epochs,
            )
            loss = losses["total"]
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = output["logits"].argmax(dim=-1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track loss components
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    loss_components[k] = loss_components.get(k, 0) + v.item()
        
        # Compute metrics
        from sklearn.metrics import f1_score, accuracy_score
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = accuracy_score(all_labels, all_preds)
        
        # Average loss components
        for k in loss_components:
            loss_components[k] /= n_batches
        
        return {
            "loss": avg_loss,
            "f1": f1,
            "accuracy": acc,
            "loss_components": loss_components,
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        use_ba: bool = False,
    ) -> Dict[str, float]:
        """Evaluate on val/test set."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in val_loader:
            f_v = batch["image_features"].to(self.device)
            f_t = batch["text_features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            output = self.model(f_v, f_t, use_ba=use_ba)
            
            # Loss (no adversarial at eval)
            losses = self.loss_fn(output, labels, None, 0)
            total_loss += losses["focal"].item()
            
            preds = output["logits"].argmax(dim=-1).cpu()
            probs = torch.softmax(output["logits"], dim=-1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.numpy())
        
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        return {
            "loss": avg_loss,
            "f1": f1,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
            "probabilities": np.array(all_probs),
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
    ) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"🚀 Training CausalCrisis V3 — {epochs} epochs")
        print(f"   Phase 1 (warmup): epochs 0-{self.warmup_epochs-1}")
        print(f"   Phase 2 (joint):  epochs {self.warmup_epochs}-{epochs-1}")
        print(f"   BA starts at:     epoch {self.ba_start_epoch}")
        print(f"{'='*60}\n")
        
        early_stop = EarlyStopping(patience=patience, mode="max")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Phase indicator
            phase = "Phase 1 (warmup)" if epoch < self.warmup_epochs else "Phase 2 (joint)"
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            use_ba = epoch >= self.ba_start_epoch
            val_metrics = self.evaluate(val_loader, use_ba=use_ba)
            
            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # History
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(current_lr)
            
            # Track adaptive weights
            if hasattr(self.loss_fn, 'adaptive_weights') and self.loss_fn.adaptive_weights is not None:
                self.history["adaptive_weights"].append(
                    self.loss_fn.adaptive_weights.get_weights()
                )
            
            # Print progress
            ba_marker = " [BA]" if use_ba else ""
            print(
                f"Epoch {epoch:3d} [{phase}]{ba_marker} | "
                f"Train: loss={train_metrics['loss']:.4f} F1={train_metrics['f1']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f} F1={val_metrics['f1']:.4f} | "
                f"LR={current_lr:.2e}"
            )
            
            # Save best model
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics)
                print(f"   ✅ Best val F1: {self.best_val_f1:.4f} (saved)")
            
            # Early stopping
            if early_stop.step(val_metrics["f1"]):
                print(f"\n⏹️ Early stopping at epoch {epoch} (patience={patience})")
                break
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"   Best val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        path = os.path.join(
            self.save_dir, 
            f"{self.experiment_name}_best.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            "history": self.history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"📦 Loaded checkpoint: epoch {ckpt['epoch']}, "
              f"F1={ckpt['metrics'].get('f1', 'N/A')}")
        return ckpt


# ============================================================================
# Generic Trainer (V4-compatible, works with any model)
# ============================================================================
class GenericTrainer:
    """
    Version-agnostic trainer — works with any model that returns logits.

    Supports:
    - 2-modal: model(f_v, f_t) → logits
    - 3-modal: model(f_v, f_t, f_llava) → logits
    - Any model returning a tensor of logits

    Example usage:
        trainer = GenericTrainer(device="cuda")
        result = trainer.run_experiment(
            "B2: GCA 3-modal", model, loaders,
            epochs=50, lr=3e-4, patience=7,
            loss_fn=FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05),
            use_llava=True, seed=42,
        )
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def train_one_epoch(self, model, loader, optimizer, loss_fn, use_llava=True):
        """Train for 1 epoch."""
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in loader:
            f_v = batch["image_features"].to(self.device)
            f_t = batch["text_features"].to(self.device)
            labels = batch["label"].to(self.device)

            f_llava = batch.get("llava_features")
            if f_llava is not None and use_llava:
                f_llava = f_llava.to(self.device)
            else:
                f_llava = None

            # Forward — model returns logits directly
            logits = model(f_v, f_t, f_llava) if f_llava is not None else model(f_v, f_t)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        return total_loss / len(loader), f1

    @torch.no_grad()
    def evaluate(self, model, loader, loss_fn, use_llava=True):
        """Evaluate model on a data loader."""
        model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            f_v = batch["image_features"].to(self.device)
            f_t = batch["text_features"].to(self.device)
            labels = batch["label"].to(self.device)

            f_llava = batch.get("llava_features")
            if f_llava is not None and use_llava:
                f_llava = f_llava.to(self.device)
            else:
                f_llava = None

            logits = model(f_v, f_t, f_llava) if f_llava is not None else model(f_v, f_t)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, -1).cpu().numpy())

        from sklearn.metrics import (
            f1_score, accuracy_score, balanced_accuracy_score,
        )

        return {
            "loss": total_loss / len(loader),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "accuracy": accuracy_score(all_labels, all_preds),
            "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
            "preds": np.array(all_preds),
            "labels": np.array(all_labels),
            "probs": np.array(all_probs),
        }

    def run_experiment(
        self,
        model_name: str,
        model: nn.Module,
        loaders: dict,
        loss_fn: nn.Module,
        epochs: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        patience: int = 7,
        use_llava: bool = True,
        seed: int = 42,
    ):
        """Run full training + evaluation experiment."""
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model.to(self.device)
        loss_fn = loss_fn.to(self.device) if hasattr(loss_fn, 'to') else loss_fn

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        best_val_f1, best_epoch, no_improve = 0, 0, 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

        for epoch in range(epochs):
            train_loss, train_f1 = self.train_one_epoch(
                model, loaders["train"], optimizer, loss_fn, use_llava
            )
            val_metrics = self.evaluate(model, loaders["val"], loss_fn, use_llava)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_metrics["f1_weighted"])

            improved = val_metrics["f1_weighted"] > best_val_f1
            if improved:
                best_val_f1 = val_metrics["f1_weighted"]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or improved:
                marker = " ✅" if improved else ""
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Train: loss={train_loss:.4f} F1={train_f1:.4f} | "
                    f"Val: F1w={val_metrics['f1_weighted']:.4f} "
                    f"F1m={val_metrics['f1_macro']:.4f}{marker}"
                )

            if no_improve >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch} (patience={patience})")
                break

        # Load best & evaluate on test
        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.to(self.device)
        test_metrics = self.evaluate(model, loaders["test"], loss_fn, use_llava)

        print(f"\n{'─' * 50}")
        print(f"  {model_name} — RESULTS (best epoch {best_epoch}):")
        print(f"  Val  F1w={best_val_f1:.4f}")
        print(f"  Test F1w={test_metrics['f1_weighted']:.4f} "
              f"F1m={test_metrics['f1_macro']:.4f} "
              f"BAcc={test_metrics['balanced_acc']:.4f}")
        print(f"{'─' * 50}")

        return {
            "model_name": model_name,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "history": history,
            "model_state": best_state,
        }


# ============================================================================
# Legacy Baseline Trainer (V3, kept for old notebooks)
# ============================================================================
class BaselineTrainer:
    """Simplified trainer cho MLP baseline (V3 legacy)."""

    def __init__(self, model, optimizer, scheduler, device="cuda", save_dir="checkpoints"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
        self.best_val_f1 = 0

    def train(self, train_loader, val_loader, epochs=100, patience=15,
              loss_fn=None):
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        early_stop = EarlyStopping(patience=patience, mode="max")

        for epoch in range(epochs):
            self.model.train()
            total_loss, all_preds, all_labels = 0, [], []

            for batch in train_loader:
                f_v = batch["image_features"].to(self.device)
                f_t = batch["text_features"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(f_v, f_t)
                loss = loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_preds.extend(logits.argmax(-1).detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if self.scheduler:
                self.scheduler.step()

            val_metrics = self._evaluate(val_loader, loss_fn)

            from sklearn.metrics import f1_score
            train_f1 = f1_score(all_labels, all_preds, average="weighted")

            self.history["train_loss"].append(total_loss / len(train_loader))
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_metrics["f1"])

            if epoch % 10 == 0 or val_metrics["f1"] > self.best_val_f1:
                print(f"Epoch {epoch:3d} | Train F1={train_f1:.4f} | Val F1={val_metrics['f1']:.4f}")

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                torch.save(self.model.state_dict(),
                          os.path.join(self.save_dir, "baseline_best.pt"))

            if early_stop.step(val_metrics["f1"]):
                print(f"Early stop at epoch {epoch}")
                break

        return self.history

    @torch.no_grad()
    def _evaluate(self, loader, loss_fn):
        self.model.eval()
        total_loss, all_preds, all_labels = 0, [], []

        for batch in loader:
            f_v = batch["image_features"].to(self.device)
            f_t = batch["text_features"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(f_v, f_t)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import f1_score, accuracy_score
        return {
            "loss": total_loss / len(loader),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "accuracy": accuracy_score(all_labels, all_preds),
        }

