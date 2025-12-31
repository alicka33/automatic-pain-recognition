import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple, Optional


class Evaluator:
    """Evaluator: holds model, test_loader, criterion and device.

    Optional configuration (model_name, num_classes, verbose) can be passed in the constructor
    so that `evaluate_epoch` does not require additional arguments and can print results
    automatically when `model_name` is provided.
    """

    def __init__(self, model: nn.Module, test_loader, criterion: Optional[nn.Module] = None, device: Optional[str] = None,
                 model_name: Optional[str] = None, num_classes: Optional[int] = None, verbose: bool = True):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

        # Optional metadata used by evaluate_epoch/print_results
        self.model_name = model_name
        self.num_classes = num_classes
        self.verbose = verbose

    def evaluate_epoch(self) -> Tuple[float, float, List[int], List[int]]:
        """Run one evaluation pass and return (avg_loss, acc, labels, preds).

        If `model_name` was provided in the constructor and `verbose` is True, this
        method will call `print_results` automatically using stored metadata.
        """
        self.model.eval()
        running_loss = 0.0
        all_labels: List[int] = []
        all_preds: List[int] = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += float(loss.item()) * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())

        total = len(self.test_loader.dataset) if hasattr(self.test_loader, "dataset") else max(len(all_labels), 1)
        avg_loss = running_loss / total if total > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

        if self.model_name is not None and self.verbose:
            try:
                self.print_results(self.model_name, avg_loss, acc, all_labels, all_preds, num_classes=self.num_classes)
            except Exception as e:
                print("Warning: failed to print results:", e)

        return avg_loss, acc, all_labels, all_preds

    def print_results(self, model_name: str, test_loss: float, test_acc: float,
                      labels: List[int], preds: List[int], num_classes: Optional[int] = None) -> None:
        """Print results + classification report (Polish headings like original)."""
        num_classes = int(num_classes) if num_classes is not None else (max(max(labels, default=0), max(preds, default=0)) + 1)
        target_names = [f'Klasa {i}' for i in range(num_classes)]

        print("\n" + "="*50)
        print(f"WYNIKI OCENY DLA MODELU: {model_name}")
        print("="*50)
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}\n')

        print("--- Szczegółowy Raport Klasyfikacji ---")
        print(classification_report(labels, preds, target_names=target_names))
        print("="*50)
        cm = confusion_matrix(labels, preds)
        print("Confusion matrix:\n", cm)
