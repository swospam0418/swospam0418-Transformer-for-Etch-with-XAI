from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def single_recipe_predict(model, seq, processor, device):
    if not seq:
        return np.zeros(len(processor.all_profile_cols), dtype=float)
    L = len(seq)
    P = max((len(s[1]) for s in seq), default=1)
    step_arr = np.zeros((L,), dtype=np.int64)
    pos_arr = np.zeros((L, P), dtype=np.int64)
    param_arr = np.zeros((L, P), dtype=np.float32)
    param_mask = np.zeros((L, P), dtype=bool)
    for i, (step, params) in enumerate(seq):
        step_arr[i] = step
        for j, (pos, val) in enumerate(params):
            pos_arr[i, j] = pos
            param_arr[i, j] = val
            param_mask[i, j] = True
    step_t = torch.from_numpy(step_arr).unsqueeze(0).to(device)
    pos_t = torch.from_numpy(pos_arr).unsqueeze(0).to(device)
    param_t = torch.from_numpy(param_arr).unsqueeze(0).to(device)
    mask_t = torch.ones((1, L), dtype=torch.bool, device=device)
    p_mask_t = torch.from_numpy(param_mask).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(step_t, pos_t, param_t, mask_t, p_mask_t)
    return out.squeeze(0).cpu().numpy()


def evaluate_predictions(model, loader: DataLoader, processor, device: str, max_samples: int = 15):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["step_seq"].to(device),
                       batch["pos_seq"].to(device),
                       batch["param_seq"].to(device),
                       batch["mask"].to(device),
                       batch["param_mask"].to(device))
            preds.append(out.cpu().numpy())
            acts.append(batch["profile"].cpu().numpy())
    preds = np.concatenate(preds)
    acts = np.concatenate(acts)
    r2 = r2_score(acts, preds)
    print(f"Overall R2: {r2:.4f}")
    preds_unscaled = np.array([processor.unscale_profile(p) for p in preds])
    acts_unscaled = np.array([processor.unscale_profile(a) for a in acts])
    for i in range(min(max_samples, len(preds))):
        plt.figure(figsize=(10,4))
        plt.plot(acts_unscaled[i], label='Actual')
        plt.plot(preds_unscaled[i], label='Pred')
        plt.title(f'Sample {i+1}')
        plt.legend()
        plt.show()
