import os
import time
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Trainer:
    """Trainer that stores model, dataloaders and training components in ctor."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        model_save_path: str = "model.pt",
        num_epochs: int = 100,
        monitor: str = "val_acc",   # 'val_acc' or 'val_loss'
        minimize_monitor: bool = False,
        save_best_only: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.model_save_path = model_save_path
        self.num_epochs = num_epochs
        self.monitor = monitor
        self.minimize_monitor = minimize_monitor
        self.save_best_only = save_best_only

        self.history: Dict[str, List[float]] = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch and return (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += float(loss.item()) * inputs.size(0)

            # accumulate predictions for train accuracy
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

        avg_loss = running_loss / len(self.train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        return avg_loss, acc

    def validate_epoch(self) -> Tuple[float, float, list, list]:
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += float(loss.item()) * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())
        avg_loss = running_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        return avg_loss, acc, all_labels, all_preds

    def _step_scheduler(self, metric_value: float):
        if self.scheduler is None:
            return
        name = self.scheduler.__class__.__name__.lower()
        # ReduceLROnPlateau requires metric; others use step()
        if "reduce" in name or "plateau" in name:
            # send metric appropriate to minimize_monitor semantics
            self.scheduler.step(metric_value if not self.minimize_monitor else metric_value)
        else:
            # epoch-based schedulers
            self.scheduler.step()

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        filename = filename or os.path.basename(self.model_save_path)
        path = os.path.join(os.path.dirname(self.model_save_path) or ".", filename)
        torch.save(self.model.state_dict(), path)
        return path

    def fit(self) -> Tuple[str, Optional[str], Dict[str, List[float]]]:
        best_metric = float('inf') if self.minimize_monitor else -float('inf')
        best_epoch = -1
        best_path = None

        print(f"Starting training for {self.num_epochs} epochs on {self.device}")

        for epoch in range(self.num_epochs):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, _, _ = self.validate_epoch()
            elapsed = time.time() - t0

            # select monitor value
            current_metric = val_loss if self.monitor == 'val_loss' else val_acc

            # scheduler update (pass a metric for ReduceLROnPlateau)
            # we use val_acc for non-minimize case by default if monitor == 'val_acc'
            sched_metric_value = val_acc if not self.minimize_monitor else val_loss
            self._step_scheduler(sched_metric_value)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            improved = (current_metric < best_metric) if self.minimize_monitor else (current_metric > best_metric)
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                if self.save_best_only:
                    best_path = self.save_checkpoint(f"best_{os.path.basename(self.model_save_path)}")
                    print(f"Saved best model to {best_path} (metric={best_metric:.4f})")

            print(f"Epoch {epoch+1}/{self.num_epochs} | Time {elapsed:.2f}s | Train loss {train_loss:.4f} | Train acc {train_acc:.4f} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f}")

        final_path = self.save_checkpoint(self.model_save_path)
        print(f"Training finished. Best epoch: {best_epoch}. Best monitored metric: {best_metric:.4f}")
        return final_path, best_path, self.history

    def print_training_config(self):
        """Print key model, optimizer and training configuration.
        Falls back gracefully when attributes are not available.
        """

        # Model information
        model_info = {"Model Type": self.model.__class__.__name__}
        lstm = getattr(self.model, 'lstm', None)
        if lstm is not None:
            model_info.update({
                "Input Features": getattr(lstm, 'input_size', 'N/A'),
                "Hidden Size": getattr(lstm, 'hidden_size', 'N/A'),
                "Num Layers": getattr(lstm, 'num_layers', 'N/A'),
                "Bidirectional": getattr(lstm, 'bidirectional', False),
                "Dropout (LSTM)": getattr(lstm, 'dropout', 'N/A'),
            })
        else:
            for attr in ('input_size', 'hidden_size', 'num_layers', 'bidirectional', 'dropout'):
                if hasattr(self.model, attr):
                    pretty = attr.replace('_', ' ').title()
                    model_info[pretty] = getattr(self.model, attr)

        model_info['Num Parameters'] = sum(p.numel() for p in self.model.parameters())

        if self.optimizer is not None:
            opt_cfg = self.optimizer.param_groups[0]
            optim_info = {
                "Optimizer": self.optimizer.__class__.__name__,
                "Learning Rate (LR)": opt_cfg.get('lr'),
                "Weight Decay (L2)": opt_cfg.get('weight_decay', 0.0),
            }
        else:
            optim_info = {"Optimizer": 'N/A'}

        batch_size = getattr(self.train_loader, 'batch_size', 'N/A')
        device = getattr(self, 'device', 'N/A')
        training_params = {
            "Batch Size": batch_size,
            "Num Epochs": self.num_epochs,
            "Device": device,
        }

        print("\n" + "="*50)
        print("🚀 TRAINING CONFIGURATION")
        print("="*50)

        print("\n--- Model Architecture ---")
        for k, v in model_info.items():
            print(f"  {k:<22}: {v}")

        print("\n--- Optimizer & Regularization ---")
        for k, v in optim_info.items():
            print(f"  {k:<22}: {v}")

        print("\n--- Training Flow ---")
        for k, v in training_params.items():
            print(f"  {k:<22}: {v}")

        print("="*50 + "\n")

    def plot_history(self, show: bool = True, save_path: Optional[str] = None):
        """Plot training history: loss (left) and accuracy (right). Optionally save to file."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='train_loss')
        plt.plot(epochs, self.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')

        plt.subplot(1, 2, 2)
        train_acc = self.history.get('train_acc')
        if train_acc is not None:
            plt.plot(epochs, train_acc, label='train_acc')
        plt.plot(epochs, self.history['val_acc'], label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()
