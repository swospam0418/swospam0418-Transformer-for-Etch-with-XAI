from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Sequence, Any


def read_merge_sheets(file_path: str) -> pd.DataFrame:
    """Read all sheets in an Excel workbook and concatenate them."""
    xls = pd.ExcelFile(file_path)
    merged = [
        pd.read_excel(file_path, sheet_name=s).assign(SHEET_NAME=s)
        for s in xls.sheet_names
    ]
    return pd.concat(merged, ignore_index=True)


class EtchDataProcessor:
    """Parse and standardize raw dataframe."""

    def __init__(self) -> None:
        self.step_types: Dict[str, int] = {}
        self.tuning_knobs: Dict[str, List[str]] = {}
        self.step_param_indices: Dict[str, List[int]] = {}
        self.all_tuning_cols: List[str] = []
        self.all_profile_cols: List[str] = []
        self.x_col_stats: Dict[str, Tuple[float, float]] = {}
        self.y_col_stats: Dict[str, Tuple[float, float]] = {}

    # --- registration -----------------------------------------------------
    def parse_columns(self, df: pd.DataFrame) -> None:
        """Register all X* and Y* columns."""
        for col in df.columns:
            if col.startswith("X"):
                parts = col.split("_")
                step_type = parts[1] if len(parts) > 1 else "UNKNOWN"
                knob = "".join(parts[2:]) if len(parts) > 2 else "default"
                if step_type not in self.step_types:
                    self.step_types[step_type] = len(self.step_types)
                    self.tuning_knobs[step_type] = []
                if knob not in self.tuning_knobs[step_type]:
                    self.tuning_knobs[step_type].append(knob)
                if col not in self.all_tuning_cols:
                    self.all_tuning_cols.append(col)
            elif col.startswith("Y") and col not in self.all_profile_cols:
                self.all_profile_cols.append(col)

    def build_indices(self) -> None:
        col_idx_map = {c: i for i, c in enumerate(self.all_tuning_cols)}
        for st, knobs in self.tuning_knobs.items():
            idxs: List[int] = []
            for knob in knobs:
                col_name = f"X{st}_{knob}" if knob != "default" else f"X{st}"
                if col_name in col_idx_map:
                    idxs.append(col_idx_map[col_name])
            self.step_param_indices[st] = idxs

    # --- statistics -------------------------------------------------------
    def fit_statistics(self, df: pd.DataFrame) -> None:
        eps = 1e-8
        for col in self.all_tuning_cols:
            if col in df.columns:
                values = df[col].dropna()
                mean, std = values.mean(), values.std()
            else:
                mean, std = 0.0, 1.0
            self.x_col_stats[col] = (float(mean), float(std or eps))
        for col in self.all_profile_cols:
            if col in df.columns:
                values = df[col].dropna()
                mean, std = values.mean(), values.std()
            else:
                mean, std = 0.0, 1.0
            self.y_col_stats[col] = (float(mean), float(std or eps))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        for col in self.all_tuning_cols:
            mean, std = self.x_col_stats.get(col, (0.0, 1.0))
            df_new[col] = ((df_new[col].fillna(mean)) - mean) / std
        for col in self.all_profile_cols:
            mean, std = self.y_col_stats.get(col, (0.0, 1.0))
            df_new[col] = ((df_new[col].fillna(mean)) - mean) / std
        return df_new

    # --- sequence construction -------------------------------------------
    def build_sequences_and_profiles(self, df: pd.DataFrame) -> Tuple[List[List[Tuple[int, np.ndarray]]], np.ndarray]:
        profiles = df[self.all_profile_cols].values
        X_matrix = df[self.all_tuning_cols].values
        param_dim = len(self.all_tuning_cols)
        sequences: List[List[Tuple[int, np.ndarray]]] = []
        for row in X_matrix:
            recipe_seq = []
            for st, st_idx in self.step_types.items():
                idxs = self.step_param_indices[st]
                if not idxs:
                    continue
                param_vec = np.zeros(param_dim, dtype=float)
                param_vec[idxs] = row[idxs]
                if np.any(param_vec):
                    recipe_seq.append((st_idx, param_vec))
            sequences.append(recipe_seq)
        return sequences, profiles

    def unscale_profile(self, profile_vec: np.ndarray) -> np.ndarray:
        unscaled = np.zeros_like(profile_vec)
        for d, col in enumerate(self.all_profile_cols):
            mean, std = self.y_col_stats.get(col, (0.0, 1.0))
            unscaled[d] = profile_vec[d] * std + mean
        return unscaled


# --- datasets ------------------------------------------------------------
class EtchDataset(Dataset):
    def __init__(self, sequences: Sequence, profiles: np.ndarray):
        self.sequences = list(sequences)
        self.profiles = profiles

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.profiles[idx]


class SingleEtchDataset(Dataset):
    def __init__(self, seq: List[Tuple[int, np.ndarray]], profile: np.ndarray):
        self.seq = seq
        self.profile = profile

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.seq, self.profile


# --- collate functions ---------------------------------------------------
def collate_fn(batch: List[Any]):
    sequences, profiles = zip(*batch)
    batch_size = len(sequences)
    max_len = max(len(s) for s in sequences) or 1
    param_dim = len(sequences[0][0][1]) if sequences[0] else 1
    step_seq = torch.zeros((batch_size, max_len), dtype=torch.long)
    param_seq = torch.zeros((batch_size, max_len, param_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        for j, (step, vec) in enumerate(seq):
            step_seq[i, j] = step
            param_seq[i, j] = torch.from_numpy(vec)
            mask[i, j] = True
    profiles_t = torch.tensor(np.stack(profiles), dtype=torch.float32)
    return {"step_seq": step_seq, "param_seq": param_seq, "mask": mask, "profile": profiles_t}


def collate_fn_single(batch: List[Any]):
    seq, profile = batch[0]
    L = max(len(seq), 1)
    param_dim = len(seq[0][1]) if seq else 1
    step_seq = torch.zeros((L,), dtype=torch.long)
    param_seq = torch.zeros((L, param_dim), dtype=torch.float32)
    mask = torch.zeros((L,), dtype=torch.bool)
    for i, (step, vec) in enumerate(seq):
        step_seq[i] = step
        param_seq[i] = torch.from_numpy(vec)
        mask[i] = True
    profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0)
    return {"step_seq": step_seq.unsqueeze(0), "param_seq": param_seq.unsqueeze(0), "mask": mask.unsqueeze(0), "profile": profile_t}

