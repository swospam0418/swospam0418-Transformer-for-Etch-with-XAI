# Transformer for Etch with XAI

This project demonstrates an encoder-only Transformer for predicting plasma
etching profiles from recipe step parameters.  The repository is organised
into a small package called `etch` consisting of several independent
modules:

- `data_processing.py` – read Excel workbooks, standardise columns and build
  PyTorch datasets.
- `model.py` – definition of the Transformer network used for prediction.
- `trainer.py` – training loop with validation and R² tracking.
- `analysis.py` – helper functions for evaluating a trained model.
- `predictor.py` – simple search utilities to find recipes that match a target
  profile.
- `sensitivity.py` – minimal tools for inspecting parameter influence.

The package exposes these utilities via ``import etch``.  A minimal example
that trains a model and runs a prediction is provided in ``main.py``:

```bash
python main.py path/to/data.xlsx --epochs 20 --batch-size 16
```

The code is intended as a clean starting point for further experimentation and
analysis of plasma etching processes.
