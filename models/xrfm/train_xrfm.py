import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from comp9417.data.data_loader import get_dataset
from xrfm import xRFM
import torch
# 如果库支持，强制使用 cpu
device = "cpu"

def evaluate_classification(y_true, y_pred, y_prob=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            result["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            result["auc"] = None
    else:
        result["auc"] = None

    return result


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"rmse": rmse}


def train_and_evaluate(dataset_name, task_type):
    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)

    print(f"Running dataset: {dataset_name}")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    if dataset_name == "diamonds":
        X_train = X_train[:10]
        y_train = y_train[:10]
        X_val = X_val[:5]
        y_val = y_val[:5]

    # 在 train_xrfm.py 中修改
    model = xRFM(use_diag=True)

    start_train = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_train

    val_pred = model.predict(X_val)

    start_infer = time.time()
    test_pred = model.predict(X_test)
    inference_time = time.time() - start_infer
    inference_time_per_sample = inference_time / len(X_test)

    if task_type == "classification":
        val_prob = None
        test_prob = None

        if hasattr(model, "predict_proba"):
            try:
                val_prob_raw = model.predict_proba(X_val)
                test_prob_raw = model.predict_proba(X_test)

                if len(val_prob_raw.shape) == 2 and val_prob_raw.shape[1] >= 2:
                    val_prob = val_prob_raw[:, 1]
                    test_prob = test_prob_raw[:, 1]
            except Exception:
                pass

        val_metrics = evaluate_classification(y_val, val_pred, val_prob)
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)

    else:
        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

    print(f"Running dataset: {dataset_name}")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print("\n=== Result ===")
    print(f"Train time: {train_time:.4f} s")
    print(f"Inference time total: {inference_time:.6f} s")
    print(f"Inference time per sample: {inference_time_per_sample:.8f} s")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")

    return {
        "dataset": dataset_name,
        "task_type": task_type,
        "train_time": train_time,
        "inference_time_total": inference_time,
        "inference_time_per_sample": inference_time_per_sample,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


if __name__ == "__main__":
    result = train_and_evaluate("diamonds", "regression")
    print(result)