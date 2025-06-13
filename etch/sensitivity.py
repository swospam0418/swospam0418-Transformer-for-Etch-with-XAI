from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

import torch

from .analysis import single_recipe_predict


class SensitivityAnalyzer:
    """Minimal sensitivity analysis utilities."""

    def __init__(self, model, processor, data: pd.DataFrame, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.processor = processor
        self.data = data
        self.device = device
        self.results: Dict[str, pd.DataFrame] = {}

    def gradient_sensitivity(self, n_samples: int = 100) -> pd.DataFrame:
        df_std = self.processor.transform(self.data)
        seqs, _ = self.processor.build_sequences_and_profiles(df_std)
        x_cols = self.processor.all_tuning_cols
        y_cols = self.processor.all_profile_cols
        mat = np.zeros((len(x_cols), len(y_cols)))
        sample_indices = np.random.choice(len(seqs), size=min(n_samples, len(seqs)), replace=False)
        for idx in sample_indices:
            seq = seqs[idx]
            if not seq:
                continue
            step_idx = np.array([s[0] for s in seq], dtype=np.int64)
            param = np.stack([s[1] for s in seq]).astype(np.float32)
            step_t = torch.from_numpy(step_idx).unsqueeze(0).to(self.device)
            param_t = torch.from_numpy(param).unsqueeze(0).to(self.device)
            param_t.requires_grad_(True)
            mask = torch.ones((1, len(seq)), dtype=torch.bool, device=self.device)
            out = self.model(step_t, param_t, mask)
            loss = out.norm()
            loss.backward()
            grad = param_t.grad.abs().sum(dim=1).cpu().numpy()
            for j in range(min(len(grad), len(x_cols))):
                mat[j] += grad[j]
        if len(sample_indices) > 0:
            mat /= len(sample_indices)
        df = pd.DataFrame(mat, index=x_cols, columns=y_cols)
        self.results['gradient'] = df
        return df

    def plot_heatmap(self, method: str = 'gradient', figsize=(12,8)) -> None:
        if method not in self.results:
            print(f"No results for method {method}")
            return
        df = self.results[method]
        plt.figure(figsize=figsize)
        sns.heatmap(df, annot=False, cmap='viridis')
        plt.title(f'Sensitivity - {method}')
        plt.xlabel('Y outputs')
        plt.ylabel('X parameters')
        plt.tight_layout()
        plt.show()
