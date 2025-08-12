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
            L = len(seq)
            P = max((len(s[1]) for s in seq), default=1)
            step_arr = np.zeros((L,), dtype=np.int64)
            pos_arr = np.zeros((L, P), dtype=np.int64)
            param_arr = np.zeros((L, P), dtype=np.float32)
            p_mask = np.zeros((L, P), dtype=bool)
            for i, (step, params) in enumerate(seq):
                step_arr[i] = step
                for j, (pos, val) in enumerate(params):
                    pos_arr[i, j] = pos
                    param_arr[i, j] = val
                    p_mask[i, j] = True
            step_t = torch.from_numpy(step_arr).unsqueeze(0).to(self.device)
            pos_t = torch.from_numpy(pos_arr).unsqueeze(0).to(self.device)
            param_t = torch.from_numpy(param_arr).unsqueeze(0).to(self.device)
            param_t.requires_grad_(True)
            mask_t = torch.ones((1, L), dtype=torch.bool, device=self.device)
            p_mask_t = torch.from_numpy(p_mask).unsqueeze(0).to(self.device)
            out = self.model(step_t, pos_t, param_t, mask_t, p_mask_t)
            loss = out.norm()
            loss.backward()
            grad_tensor = param_t.grad.abs()[0].cpu().numpy()
            pos_np = pos_arr
            mask_np = p_mask
            grad_vec = np.zeros(len(x_cols))
            for i in range(L):
                for j in range(P):
                    if mask_np[i, j]:
                        grad_vec[pos_np[i, j]] += grad_tensor[i, j]
            mat += grad_vec[:, None]
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
