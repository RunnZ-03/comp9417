import os
import sys
import json
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset
from comp9417.models.mlp.train_mlp import _train_mlp, evaluate_regression


# =========================
# CONFIG
# =========================
DATASETS = [
    ("diamonds", "regression"),
    ("superconductivity", "regression"),
]

SUBSAMPLE_SIZES = [1000, 5000, 10000, 30000]


# =========================
# MAIN
# =========================
def run_scaling_experiment():

    all_results = []

    for dataset_name, task_type in DATASETS:

        print(f"\n{'='*60}")
        print(f"Scaling on: {dataset_name}")
        print(f"{'='*60}")

        X_train, X_val, X_test, y_train, y_val, y_test, _ = get_dataset(dataset_name)

        for n in SUBSAMPLE_SIZES:

            if n > len(X_train):
                continue

            print(f"\n--- n = {n} ---")

            np.random.seed(42)
            idx = np.random.choice(len(X_train), n, replace=False)

            X_sub = X_train[idx]
            y_sub = y_train[idx]

            # =========================
            # Train
            # =========================
            model, _, test_pred, train_time, infer_time = _train_mlp(
                X_sub, y_sub,
                X_val, y_val,
                X_test,
                task_type
            )

            # =========================
            # Evaluate
            # =========================
            test_metrics = evaluate_regression(y_test, test_pred)

            result = {
                "dataset": dataset_name,
                "n_train": n,
                "rmse": test_metrics["rmse"],
                "train_time": train_time,
                "infer_time_per_sample": infer_time / len(X_test),
            }

            all_results.append(result)

            print(f"RMSE: {result['rmse']:.4f}")
            print(f"Train time: {train_time:.2f}s")

    # =========================
    # Save
    # =========================
    save_path = os.path.join(
        PROJECT_PARENT,
        "comp9417",
        "results",
        "mlp_scaling_results.json"
    )

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    run_scaling_experiment()