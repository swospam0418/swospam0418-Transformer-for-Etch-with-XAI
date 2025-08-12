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
        # keep raw sequences for compatibility
        self.sequences = list(sequences)
        self.profiles = profiles
        self.param_dim = len(self.sequences[0][0][1]) if self.sequences and self.sequences[0] else 1

        def _prep(seq: List[Tuple[int, np.ndarray]]) -> Dict[str, np.ndarray]:
            L = len(seq)
            step_seq = np.array([s for s, _ in seq], dtype=np.int64) if L else np.zeros(0, dtype=np.int64)
            pos_seq = np.arange(L, dtype=np.int64) if L else np.zeros(0, dtype=np.int64)
            if L:
                param_seq = np.stack([v for _, v in seq]).astype(np.float32)
                param_mask = (param_seq != 0).astype(np.bool_)
            else:
                param_seq = np.zeros((0, self.param_dim), dtype=np.float32)
                param_mask = np.zeros((0, self.param_dim), dtype=np.bool_)
            mask = np.ones(L, dtype=np.bool_)
            return {
                "step_seq": step_seq,
                "pos_seq": pos_seq,
                "param_seq": param_seq,
                "param_mask": param_mask,
                "mask": mask,
            }

        self._seq_data = [_prep(seq) for seq in self.sequences]

    def __len__(self) -> int:
        return len(self._seq_data)

    def __getitem__(self, idx: int):
        return self._seq_data[idx], self.profiles[idx]


class SingleEtchDataset(Dataset):
    def __init__(self, seq: List[Tuple[int, np.ndarray]], profile: np.ndarray):
        param_dim = len(seq[0][1]) if seq else 1
        L = len(seq)
        step_seq = np.array([s for s, _ in seq], dtype=np.int64) if L else np.zeros(0, dtype=np.int64)
        pos_seq = np.arange(L, dtype=np.int64) if L else np.zeros(0, dtype=np.int64)
        if L:
            param_seq = np.stack([v for _, v in seq]).astype(np.float32)
            param_mask = (param_seq != 0).astype(np.bool_)
        else:
            param_seq = np.zeros((0, param_dim), dtype=np.float32)
            param_mask = np.zeros((0, param_dim), dtype=np.bool_)
        mask = np.ones(L, dtype=np.bool_)
        self.sample = {
            "step_seq": step_seq,
            "pos_seq": pos_seq,
            "param_seq": param_seq,
            "param_mask": param_mask,
            "mask": mask,
        }
        self.profile = profile

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample, self.profile


# --- collate functions ---------------------------------------------------
def collate_fn(batch: List[Any]):
    sequences, profiles = zip(*batch)
    batch_size = len(sequences)
    max_len = max(s["step_seq"].shape[0] for s in sequences) or 1
    param_dim = sequences[0]["param_seq"].shape[1] if sequences[0]["param_seq"].size else 1
    step_seq = torch.zeros((batch_size, max_len), dtype=torch.long)
    pos_seq = torch.zeros((batch_size, max_len), dtype=torch.long)
    param_seq = torch.zeros((batch_size, max_len, param_dim), dtype=torch.float32)
    param_mask = torch.zeros((batch_size, max_len, param_dim), dtype=torch.bool)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        L = seq["step_seq"].shape[0]
        step_seq[i, :L] = torch.from_numpy(seq["step_seq"])
        pos_seq[i, :L] = torch.from_numpy(seq["pos_seq"])
        param_seq[i, :L] = torch.from_numpy(seq["param_seq"])
        param_mask[i, :L] = torch.from_numpy(seq["param_mask"])
        mask[i, :L] = torch.from_numpy(seq["mask"])
    profiles_t = torch.tensor(np.stack(profiles), dtype=torch.float32)
    return {
        "step_seq": step_seq,
        "pos_seq": pos_seq,
        "param_seq": param_seq,
        "param_mask": param_mask,
        "mask": mask,
        "profile": profiles_t,
    }


def collate_fn_single(batch: List[Any]):
    seq, profile = batch[0]
    step_t = torch.from_numpy(seq["step_seq"]).unsqueeze(0).long()
    pos_t = torch.from_numpy(seq["pos_seq"]).unsqueeze(0).long()
    param_t = torch.from_numpy(seq["param_seq"]).unsqueeze(0).float()
    param_mask_t = torch.from_numpy(seq["param_mask"]).unsqueeze(0).bool()
    mask_t = torch.from_numpy(seq["mask"]).unsqueeze(0).bool()
    profile_t = torch.tensor(profile, dtype=torch.float32).unsqueeze(0)
    return {
        "step_seq": step_t,
        "pos_seq": pos_t,
        "param_seq": param_t,
        "param_mask": param_mask_t,
        "mask": mask_t,
        "profile": profile_t,
    }

