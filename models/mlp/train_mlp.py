import os
import sys
import json
import time
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

# =========================
# Path setup
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset

np.random.seed(42)

RESULTS_DIR = os.path.join(PROJECT_PARENT, "comp9417", "results")


# =========================
# Evaluation (UNCHANGED)
# =========================
def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if y_prob is not None:
        try:
            result["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["auc"] = None
    else:
        result["auc"] = None
    return result


def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse}


# =========================
# MLP training
# =========================
def _train_mlp(X_train, y_train, X_val, y_val, X_test, task_type):

    # ===== scaling =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ===== model =====
    if task_type == "classification":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )

    # ===== train =====
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # ===== val =====
    val_pred = model.predict(X_val)

    # ===== test =====
    t1 = time.time()
    test_pred = model.predict(X_test)
    infer_time = time.time() - t1

    return model, val_pred, test_pred, train_time, infer_time


# =========================
# MAIN PIPELINE (MLP)
# =========================
def train_and_evaluate(model_name: str, dataset_name: str, task_type: str) -> Dict[str, Any]:

    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)

    # ===== label encoding =====
    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    if task_type == "classification" and y_train.dtype.kind not in ("i", "u", "b"):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val   = le.transform(y_val)
        y_test  = le.transform(y_test)

    print(f"\n--- MLP | {dataset_name} | {task_type} ---")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # ===== train =====
    model, val_pred, test_pred, train_time, infer_time_total = _train_mlp(
        X_train, y_train, X_val, y_val, X_test, task_type
    )

    infer_time_per_sample = infer_time_total / max(len(X_test), 1)

    # =========================
    # evaluation
    # =========================
    if task_type == "classification":

        if hasattr(model, "predict_proba"):
            try:
                val_prob = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                val_prob = None
                test_prob = None
        else:
            val_prob = None
            test_prob = None

        val_metrics = evaluate_classification(y_val, val_pred, val_prob)
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)

    else:
        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

    # =========================
    # result format
    # =========================
    result = {
        "model": "mlp",
        "dataset": dataset_name,
        "task_type": task_type,

        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),

        "train_time": float(train_time),
        "infer_time_total": float(infer_time_total),
        "infer_time_per_sample": float(infer_time_per_sample),

        "val_metrics": val_metrics,
        "test_metrics": test_metrics,

        "feature_importance_top5": None,
    }

    # ===== save =====
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"mlp_{dataset_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Train time: {train_time:.4f}s")
    print(f"Infer time: {infer_time_total:.6f}s")
    print(f"Test metrics: {test_metrics}")
    print(f"Saved → {save_path}")

    return result


# =========================
# standalone test
# =========================
if __name__ == "__main__":
    result = train_and_evaluate("mlp", "diamonds", "regression")
    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))