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
            
            # Loss
            losses = self.loss_fn(output, labels, domain_ids_input, epoch)
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
# Baseline MLP Trainer (simpler, for H1)
# ============================================================================
class BaselineTrainer:
    """Simplified trainer cho MLP baseline (H1)."""
    
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
        """Train baseline model."""
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        early_stop = EarlyStopping(patience=patience, mode="max")
        
        for epoch in range(epochs):
            # Train
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
            
            # Eval
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
