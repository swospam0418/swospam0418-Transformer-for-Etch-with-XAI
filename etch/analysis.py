from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from .data_processing import SingleEtchDataset, collate_fn_single


def single_recipe_predict(model, seq, processor, device):
    if not seq:
        return np.zeros(len(processor.all_profile_cols), dtype=float)
    step_seq = np.array([s[0] for s in seq], dtype=np.int64)
    param_seq = np.stack([s[1] for s in seq]).astype(np.float32)
    step_t = torch.from_numpy(step_seq).unsqueeze(0).to(device)
    param_t = torch.from_numpy(param_seq).unsqueeze(0).to(device)
    mask = torch.ones((1, len(seq)), dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        out = model(step_t, param_t, mask)
    return out.squeeze(0).cpu().numpy()


def evaluate_predictions(model, loader: DataLoader, processor, device: str, max_samples: int = 15):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["step_seq"].to(device), batch["param_seq"].to(device), batch["mask"].to(device))
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
