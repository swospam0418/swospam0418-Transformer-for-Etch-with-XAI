from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Sequence, Any
import re


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
        self.tuning_cols: Dict[str, Dict[int, List[str]]] = {}
        self.step_param_indices: Dict[str, List[int]] = {}
        self.step_position_param_indices: Dict[str, Dict[int, List[int]]] = {}
        self.step_param_masks: Dict[str, Dict[int, np.ndarray]] = {}
        self.all_tuning_cols: List[str] = []
        self.all_profile_cols: List[str] = []
        self.x_col_stats: Dict[str, Tuple[float, float]] = {}
        self.y_col_stats: Dict[str, Tuple[float, float]] = {}
        self.global_param_dim: int = 0

    # --- registration -----------------------------------------------------
    def parse_columns(self, df: pd.DataFrame) -> None:
        """Register all X* and Y* columns."""
        for col in df.columns:
            if col.startswith("X"):
                parts = col.split("_")
                step_raw = parts[1] if len(parts) > 1 else "UNKNOWN"
                step_type, position = self.parse_architecture_step_number(step_raw)
                if step_type not in self.step_types:
                    self.step_types[step_type] = len(self.step_types)
                    self.tuning_cols[step_type] = {}
                if position not in self.tuning_cols[step_type]:
                    self.tuning_cols[step_type][position] = []
                if col not in self.tuning_cols[step_type][position]:
                    self.tuning_cols[step_type][position].append(col)
                if col not in self.all_tuning_cols:
                    self.all_tuning_cols.append(col)
            elif col.startswith("Y") and col not in self.all_profile_cols:
                self.all_profile_cols.append(col)

    def build_indices(self) -> None:
        col_idx_map = {c: i for i, c in enumerate(self.all_tuning_cols)}
        self.global_param_dim = len(self.all_tuning_cols)
        for st, pos_cols in self.tuning_cols.items():
            union_idxs: List[int] = []
            self.step_position_param_indices[st] = {}
            self.step_param_masks[st] = {}
            for pos, cols in pos_cols.items():
                idxs = [col_idx_map[c] for c in cols if c in col_idx_map]
                self.step_position_param_indices[st][pos] = idxs
                mask = np.zeros(self.global_param_dim, dtype=bool)
                mask[idxs] = True
                self.step_param_masks[st][pos] = mask
                union_idxs.extend(idxs)
            self.step_param_indices[st] = sorted(set(union_idxs))

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

    def parse_step_order(self, step_identifier: str) -> int:
        """Extract step order from an identifier."""
        m = re.search(r"(\d+)$", step_identifier)
        return int(m.group(1)) if m else 0

    def parse_architecture_step_number(self, step_identifier: str) -> Tuple[str, int]:
        """Split identifier into base step name and position."""
        m = re.match(r"([A-Za-z]+)(\d+)$", step_identifier)
        if m:
            return m.group(1), int(m.group(2))
        return step_identifier, 0

    def build_sequences_and_profiles_v2(
        self, df: pd.DataFrame
    ) -> Tuple[List[List[Tuple[int, np.ndarray, int, np.ndarray]]], np.ndarray]:
        """Return sequences of (step_type, params, position, param_mask)."""
        profiles = df[self.all_profile_cols].values
        X_matrix = df[self.all_tuning_cols].values
        sequences: List[List[Tuple[int, np.ndarray, int, np.ndarray]]] = []
        for row in X_matrix:
            recipe_seq: List[Tuple[int, np.ndarray, int, np.ndarray]] = []
            for st, st_idx in self.step_types.items():
                pos_map = self.step_position_param_indices.get(st, {})
                mask_map = self.step_param_masks.get(st, {})
                for pos, idxs in pos_map.items():
                    if not idxs:
                        continue
                    values = row[idxs]
                    if np.any(values):
                        param_vec = np.zeros(self.global_param_dim, dtype=float)
                        param_vec[idxs] = values
                        mask = mask_map[pos].copy()
                        recipe_seq.append((st_idx, param_vec, pos, mask))
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

