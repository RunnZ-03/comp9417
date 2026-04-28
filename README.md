# COMP9417 Group Project: xRFM for Tabular Data

Replication and evaluation of **xRFM** (Beaglehole et al., 2025) against
XGBoost, LightGBM, and MLP across five tabular datasets.

---

## Requirements

```bash
pip install xrfm xgboost lightgbm scikit-learn numpy pandas jinja2
```

Python 3.9+ is required.  All random seeds are fixed at 42 for reproducibility.

---

## Project Structure

```
comp9417/
├── data/
│   └── data_loader.py          # Dataset loading, preprocessing, splitting
├── models/
│   ├── xrfm/
│   │   ├── train_xrfm.py       # xRFM training & evaluation
│   │   └── run_all_xrfm.py     # Run all xRFM experiments + scaling
│   ├── xgboost/
│   │   ├── train_baselines.py  # XGBoost & LightGBM training
│   │   ├── run_all_baselines.py# Run all baseline experiments
│   │   └── run_scaling.py      # XGBoost & LightGBM scaling experiments
│   └── mlp/
│       ├── train_mlp.py        # MLP training & evaluation
│       ├── run_mlp.py          # Run all MLP experiments
│       └── run_scaling_mlp.py  # MLP scaling experiments
├── bonus/
│   └── agop_split.py           # Bonus: AGOP splitting criterion from scratch
├── utils/
│   ├── metrics.py              # Evaluation metric helpers
│   ├── timer.py                # Training/inference timing helpers
│   └── plot_utils.py           # Scalability curve plotting
├── results/                    # JSON result files (auto-generated)
├── report/                     # LaTeX report source
├── result_summary.py           # Generate summary CSV and LaTeX tables
└── README.md
```

---

## How to Run

### 1. Run all experiments

Each model can be run independently.  Results are saved to `results/` as JSON.

```bash
# xRFM (all 5 datasets + scaling)
python models/xrfm/run_all_xrfm.py

# XGBoost & LightGBM (all 5 datasets)
python models/xgboost/run_all_baselines.py

# XGBoost & LightGBM scaling experiments
python models/xgboost/run_scaling.py

# MLP (all 5 datasets)
python models/mlp/run_mlp.py

# MLP scaling experiments
python models/mlp/run_scaling_mlp.py
```

### 2. Generate summary tables

Produces `results/summary_all_results.csv` and two LaTeX tables:

```bash
python result_summary.py
```

### 3. Bonus: AGOP splitting criterion

Runs the scratch implementation and verifies agreement with xRFM library:

```bash
python bonus/agop_split.py
```

Expected output: top-2 feature overlap 2/2, cosine similarity ≈ 0.9985.

---

## Datasets

| Dataset | Task | n (train) | d | Source |
|---------|------|-----------|---|--------|
| Diamonds | Regression | 32,365 | 26 | Kaggle |
| Superconductivity | Regression | 12,757 | 81 | UCI |
| Online Shoppers | Classification | 7,398 | 27 | UCI |
| Stroke Prediction | Classification | 3,066 | 21 | Kaggle |
| HR Attrition | Classification | 882 | 51 | IBM/Kaggle |

All datasets use a 60/20/20 train/val/test split with seed 42.
Numerical features are z-score normalised; categorical features are one-hot encoded.

---

## Results Summary

Key results (test set):

| Dataset | Metric | xRFM | XGBoost | LightGBM | MLP |
|---------|--------|------|---------|----------|-----|
| Diamonds | RMSE ↓ | 1496.7 | 544.7 | **536.0** | 734.7 |
| Superconductivity | RMSE ↓ | 9.45 | 9.30 | **9.21** | 11.07 |
| Online Shoppers | AUC ↑ | 0.902 | **0.931** | **0.931** | 0.904 |
| Stroke | AUC ↑ | 0.761 | **0.839** | 0.823 | 0.269 |
| HR Attrition | Accuracy ↑ | **0.867** | 0.857 | 0.847 | 0.864 |
