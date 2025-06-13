from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from etch.data_processing import read_merge_sheets, EtchDataProcessor, EtchDataset, collate_fn
from etch.model import EtchTransformer
from etch.trainer import ModelTrainer
from etch.analysis import evaluate_predictions
from etch.predictor import ForwardPredictor


def build_datasets(excel_path: str, processor: EtchDataProcessor, test_ratio: float = 0.2):
    df = read_merge_sheets(excel_path)
    processor.parse_columns(df)
    processor.build_indices()
    processor.fit_statistics(df)
    df_std = processor.transform(df)
    sequences, profiles = processor.build_sequences_and_profiles(df_std)
    idx_all = np.arange(len(sequences))
    np.random.shuffle(idx_all)
    split = int(len(idx_all) * (1 - test_ratio))
    train_idx, val_idx = idx_all[:split], idx_all[split:]
    train_ds = EtchDataset([sequences[i] for i in train_idx], profiles[train_idx])
    val_ds = EtchDataset([sequences[i] for i in val_idx], profiles[val_idx])
    return train_ds, val_ds, processor, df


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('excel', help='Excel file path')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    opts = p.parse_args(args)
    proc = EtchDataProcessor()
    train_ds, val_ds, proc, df = build_datasets(opts.excel, proc)
    seq_lengths = [len(s) for s in train_ds.sequences]
    model = EtchTransformer(num_step_types=len(proc.step_types),
                            param_dim=len(proc.all_tuning_cols),
                            profile_dim=len(proc.all_profile_cols),
                            max_seq_len=max(seq_lengths) + 5)
    trainer = ModelTrainer(model)
    trainer.train_and_validate(train_ds, val_ds, batch_size=opts.batch_size, epochs=opts.epochs)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)
    evaluate_predictions(model, val_loader, proc, trainer.device, max_samples=3)

    # --- forward prediction demo -------------------------------------
    predictor = ForwardPredictor(model, proc, df, device=trainer.device)
    target_dict = {proc.all_profile_cols[0]: 0.0}
    target_profile = predictor.create_target_profile(target_dict)
    best_seq, err = predictor.random_search(target_profile, n_iter=100)
    if best_seq is not None:
        print("Best recipe error", err)
        for step in predictor.format_recipe(best_seq):
            print(step)


if __name__ == '__main__':
    main()
