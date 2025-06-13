from __future__ import annotations
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from .data_processing import EtchDataProcessor
from .analysis import single_recipe_predict


class ForwardPredictor:
    """Search for recipes that achieve target Y profiles."""

    def __init__(self, model: nn.Module, processor: EtchDataProcessor,
                 data, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        df_std = processor.transform(data)
        self.seqs, self.profiles = processor.build_sequences_and_profiles(df_std)

    # ------------------------------------------------------------------
    def create_target_profile(self, target_dict: Dict[str, float],
                              default: float = 0.0) -> np.ndarray:
        t = np.full(len(self.processor.all_profile_cols), default, dtype=float)
        for k, v in (target_dict or {}).items():
            if k in self.processor.all_profile_cols:
                idx = self.processor.all_profile_cols.index(k)
                mean, std = self.processor.y_col_stats[k]
                t[idx] = (v - mean) / std
        return t

    def evaluate_recipe(self, seq: Sequence[Tuple[int, np.ndarray]], target: np.ndarray) -> float:
        pred = single_recipe_predict(self.model, list(seq), self.processor, self.device)
        return float(np.mean((pred - target) ** 2))

    # --- simple random search ---------------------------------------
    def random_search(self, target: np.ndarray, n_iter: int = 200) -> Tuple[List[Tuple[int, np.ndarray]], float]:
        best_seq, best_err = None, float('inf')
        for _ in range(n_iter):
            seq = deepcopy(random.choice(self.seqs))
            err = self.evaluate_recipe(seq, target)
            if err < best_err:
                best_seq, best_err = seq, err
        return best_seq, best_err

    def format_recipe(self, seq: Sequence[Tuple[int, np.ndarray]]) -> List[Dict]:
        out = []
        for step_idx, (stype, vec) in enumerate(seq):
            step_name = [k for k, v in self.processor.step_types.items() if v == stype]
            step_name = step_name[0] if step_name else str(stype)
            params = {}
            for i, val in enumerate(vec):
                if abs(val) > 1e-3:
                    name = self.processor.all_tuning_cols[i]
                    mean, std = self.processor.x_col_stats[name]
                    params[name] = val * std + mean
            out.append({"step": step_idx + 1, "step_type": step_name, "parameters": params})
        return out

