import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from litecascade.eval.metrics import bootstrap_metrics, compute_detection_metrics
from litecascade.models.baselines import B2_Trees

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Trainer handling PyTorch and Tree-based baselines."""

    def __init__(self, model_name: str, model: Any, device: str = "cpu", run_dir: str = "results/default", seed: int = 42):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.is_tree = isinstance(model, B2_Trees)

        if not self.is_tree:
            self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3,
        class_weights: torch.Tensor = None,
        use_amp: bool = False,
    ):
        if self.is_tree:
            return self._fit_tree(train_loader, val_loader)

        return self._fit_pytorch(train_loader, val_loader, epochs, lr, class_weights, use_amp)

    def _fit_tree(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(f"Fitting Tree-based model: {self.model_name}")
        # Extract all data
        X_train, y_train = self._extract_data(train_loader)
        X_val, y_val = self._extract_data(val_loader)

        # Flatten windows for tree models: (N, T, F) -> (N, T*F)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        self.model.fit(X_train_flat, y_train)

        val_preds = self.model.predict(X_val_flat)
        metrics = compute_detection_metrics(y_val, val_preds)
        logger.info(f"Tree Model Val F1: {metrics['f1_macro']:.4f}")
        return metrics

    def _fit_pytorch(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, class_weights: torch.Tensor, use_amp: bool
    ):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Optional L1 regularization for B1 (SNIPE)
        l1_lambda = 1e-5 if self.model_name.lower() == "b1" else 0.0

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_f1 = -1.0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(X)
                    loss = criterion(logits, y)

                    if l1_lambda > 0:
                        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                        loss = loss + l1_lambda * l1_norm

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            scheduler.step()
            train_loss /= len(train_loader)

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics["f1_macro"]

            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, self.run_dir / "best_model.pt")

        return best_f1

    def evaluate(self, loader: DataLoader, bootstrap: bool = False) -> dict[str, Any]:
        if self.is_tree:
            X, y = self._extract_data(loader)
            X_flat = X.reshape(X.shape[0], -1)
            preds = self.model.predict(X_flat)
            metrics = compute_detection_metrics(y, preds)
            if bootstrap:
                ci = bootstrap_metrics(y, preds)
                metrics.update(ci)
            return metrics

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        if len(y_true) == 0:
            logger.warning("Empty evaluation set.")
            return {"f1_macro": 0.0, "accuracy": 0.0, "precision": 0.0, "recall_malicious": 0.0, "fpr": 0.0, "mcc": 0.0}

        metrics = compute_detection_metrics(y_true, y_pred)
        if bootstrap:
            ci = bootstrap_metrics(y_true, y_pred)
            metrics.update(ci)

        return metrics

    def _extract_data(self, loader: DataLoader):
        X_list, y_list = [], []
        for X, y in loader:
            X_list.append(X.numpy())
            y_list.append(y.numpy())
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
