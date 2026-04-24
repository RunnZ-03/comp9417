"""
run_scaling.py
==============
Scalability experiment: train XGBoost and LightGBM on subsampled training sets
of diamonds and superconductivity, recording test RMSE and training time at each
sample size. Results are saved to results/xgboost_scaling_results.json and
results/lightgbm_scaling_results.json for use in the scalability plot.
"""

import os
import sys
import json
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset_subsampled

np.random.seed(42)

RESULTS_DIR = os.path.join(PROJECT_PARENT, "comp9417", "results")

DATASETS   = ["diamonds", "superconductivity"]
SAMPLE_SIZES = [1000, 5000, 10000, 30000]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_xgboost_scaling():
    import xgboost as xgb

    records = []
    for dataset in DATASETS:
        for n in SAMPLE_SIZES:
            X_train, X_val, X_test, y_train, y_val, y_test, _ = get_dataset_subsampled(
                dataset, sample_size_absolute=n
            )
            actual_n = len(X_train)

            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=20,
            )

            t0 = time.time()
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            train_time = time.time() - t0

            t1 = time.time()
            test_pred = model.predict(X_test)
            infer_time_per_sample = (time.time() - t1) / max(len(X_test), 1)

            test_rmse = rmse(y_test, test_pred)

            rec = {
                "dataset": dataset,
                "n_train": actual_n,
                "rmse": test_rmse,
                "train_time": train_time,
                "infer_time_per_sample": infer_time_per_sample,
            }
            records.append(rec)
            print(f"  XGBoost | {dataset} | n={actual_n:>6} | RMSE={test_rmse:.4f} | t={train_time:.3f}s")

    out_path = os.path.join(RESULTS_DIR, "xgboost_scaling_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved → {out_path}\n")
    return records


def run_lightgbm_scaling():
    import lightgbm as lgb

    records = []
    for dataset in DATASETS:
        for n in SAMPLE_SIZES:
            X_train, X_val, X_test, y_train, y_val, y_test, _ = get_dataset_subsampled(
                dataset, sample_size_absolute=n
            )
            actual_n = len(X_train)

            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

            t0 = time.time()
            try:
                callbacks = [
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            except TypeError:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          early_stopping_rounds=20, verbose=False)
            train_time = time.time() - t0

            t1 = time.time()
            test_pred = model.predict(X_test)
            infer_time_per_sample = (time.time() - t1) / max(len(X_test), 1)

            test_rmse = rmse(y_test, test_pred)

            rec = {
                "dataset": dataset,
                "n_train": actual_n,
                "rmse": test_rmse,
                "train_time": train_time,
                "infer_time_per_sample": infer_time_per_sample,
            }
            records.append(rec)
            print(f"  LightGBM | {dataset} | n={actual_n:>6} | RMSE={test_rmse:.4f} | t={train_time:.3f}s")

    out_path = os.path.join(RESULTS_DIR, "lightgbm_scaling_results.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved → {out_path}\n")
    return records


if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Scaling Experiment")
    print("=" * 60)
    run_xgboost_scaling()

    print("=" * 60)
    print("LightGBM Scaling Experiment")
    print("=" * 60)
    run_lightgbm_scaling()

    print("All scaling experiments done.")
