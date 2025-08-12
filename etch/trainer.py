from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from typing import List

from .data_processing import collate_fn


class ModelTrainer:
    """Utility for training the transformer model."""

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_r2s: List[float] = []
        self.val_r2s: List[float] = []

    def train_and_validate(self, train_ds: Dataset, val_ds: Dataset,
                           batch_size: int = 32, epochs: int = 100,
                           lr: float = 1e-3, weight_decay: float = 1e-4) -> None:
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                        epochs=epochs, steps_per_epoch=len(train_loader))
        criterion = nn.SmoothL1Loss(beta=1.0)
        for epoch in range(1, epochs + 1):
            self.model.train()
            tot_loss = 0.0
            preds, trues = [], []
            for batch in train_loader:
                optimizer.zero_grad()
                out = self.model(batch["step_seq"].to(self.device),
                                 batch["pos_seq"].to(self.device),
                                 batch["param_seq"].to(self.device),
                                 batch["mask"].to(self.device),
                                 batch["param_mask"].to(self.device))
                tgt = batch["profile"].to(self.device)
                loss = criterion(out, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tot_loss += loss.item()
                preds.append(out.detach().cpu().numpy())
                trues.append(tgt.cpu().numpy())
            train_loss = tot_loss / len(train_loader)
            train_r2 = r2_score(np.concatenate(trues), np.concatenate(preds))
            self.model.eval()
            val_loss, val_preds, val_trues = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = self.model(batch["step_seq"].to(self.device),
                                     batch["pos_seq"].to(self.device),
                                     batch["param_seq"].to(self.device),
                                     batch["mask"].to(self.device),
                                     batch["param_mask"].to(self.device))
                    tgt = batch["profile"].to(self.device)
                    loss = criterion(out, tgt)
                    val_loss += loss.item()
                    val_preds.append(out.cpu().numpy())
                    val_trues.append(tgt.cpu().numpy())
            val_loss /= len(val_loader)
            val_r2 = r2_score(np.concatenate(val_trues), np.concatenate(val_preds))
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_r2s.append(train_r2)
            self.val_r2s.append(val_r2)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"[Epoch {epoch:3d}/{epochs}] Loss={train_loss:.4f}/{val_loss:.4f} R2={train_r2:.3f}/{val_r2:.3f}")
