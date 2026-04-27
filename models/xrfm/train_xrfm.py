import os
import sys
import json
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset, get_dataset_subsampled
from xrfm import xRFM


np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

    if y_prob is not None:
        try:
            result["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result["auc"] = None
    else:
        result["auc"] = None

    return result


def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(rmse)
    }


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    if hasattr(x, "tolist"):
        return np.array(x.tolist())
    return np.array(x)


def build_top5_from_values(values, features):
    values = to_numpy(values)

    if values is None:
        return None

    values = np.array(values)

    if values.ndim == 0:
        return None

    if values.ndim == 2:
        if values.shape[0] == values.shape[1]:
            values = np.diag(values)
        else:
            values = values.reshape(-1)

    if values.ndim >= 1:
        values = values.reshape(-1)

    if values.size == 0:
        return None

    if values.size != len(features):
        return None

    pairs = list(zip(features, values.tolist()))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return pairs[:5]


def extract_feature_importance(model, features):
    try:
        if hasattr(model, "trees") and isinstance(model.trees, list) and len(model.trees) > 0:
            tree0 = model.trees[0]

            if isinstance(tree0, dict) and "model" in tree0:
                inner_model = tree0["model"]

                if hasattr(inner_model, "M"):
                    result = build_top5_from_values(getattr(inner_model, "M"), features)
                    if result is not None:
                        return result

                if hasattr(inner_model, "agop_best_model"):
                    agop_obj = getattr(inner_model, "agop_best_model")

                    if hasattr(agop_obj, "M"):
                        result = build_top5_from_values(getattr(agop_obj, "M"), features)
                        if result is not None:
                            return result

                    if hasattr(agop_obj, "diag"):
                        result = build_top5_from_values(getattr(agop_obj, "diag"), features)
                        if result is not None:
                            return result

                if hasattr(inner_model, "diag"):
                    result = build_top5_from_values(getattr(inner_model, "diag"), features)
                    if result is not None:
                        return result

        return "Could not extract AGOP from model.trees[0]['model']"

    except Exception as e:
        return f"Could not extract AGOP: {repr(e)}"


def load_dataset(dataset_name: str, sample_size_absolute: Optional[int] = None):
    if sample_size_absolute is not None:
        return get_dataset_subsampled(
            dataset_name,
            sample_size_absolute=sample_size_absolute
        )

    return get_dataset(dataset_name)


def get_classification_scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            prob_raw = to_numpy(model.predict_proba(X))

            if prob_raw is not None:
                prob_raw = np.array(prob_raw)

                if prob_raw.ndim == 2 and prob_raw.shape[1] >= 2:
                    return prob_raw[:, 1]

                if prob_raw.ndim == 1:
                    return prob_raw

        except Exception as e:
            print(f"Could not get predict_proba: {repr(e)}")

    if hasattr(model, "decision_function"):
        try:
            scores = to_numpy(model.decision_function(X))

            if scores is not None:
                scores = np.array(scores)

                if scores.ndim == 2 and scores.shape[1] >= 2:
                    return scores[:, 1]

                if scores.ndim == 1:
                    return scores

        except Exception as e:
            print(f"Could not get decision_function: {repr(e)}")

    return None


def train_and_evaluate(
    dataset_name: str,
    task_type: str,
    sample_size_absolute: Optional[int] = None,
    save_result: bool = True
) -> Dict[str, Any]:

    X_train, X_val, X_test, y_train, y_val, y_test, features = load_dataset(
        dataset_name,
        sample_size_absolute=sample_size_absolute
    )

    print(f"\n--- Running dataset: {dataset_name} ---")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Device: {DEVICE}")

    if sample_size_absolute is not None:
        print(f"Subsample size: {sample_size_absolute}")

    model = xRFM(use_diag=True, device=DEVICE)
    print(model)

    try:
        if DEVICE == "cuda" and hasattr(model, "rfm_params"):
            if "fit" in model.rfm_params:
                model.rfm_params["fit"]["n_iter"] = 3
                model.rfm_params["fit"]["M_batch_size"] = 128
                print("GPU settings applied: n_iter=3, M_batch_size=128")
    except Exception as e:
        print(f"Skipped rfm_params adjustment: {repr(e)}")

    start_train = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_train

    val_pred = model.predict(X_val)

    start_infer = time.time()
    test_pred = model.predict(X_test)
    infer_time_total = time.time() - start_infer
    infer_time_per_sample = infer_time_total / max(len(X_test), 1)

    if task_type == "classification":
        val_prob = get_classification_scores(model, X_val)
        test_prob = get_classification_scores(model, X_test)

        val_metrics = evaluate_classification(y_val, val_pred, val_prob)
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)

    else:
        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

    feature_importance_top5 = extract_feature_importance(model, features)

    result_dict = {
        "model": "xrfm",
        "dataset": dataset_name,
        "task_type": task_type,
        "sample_size_absolute": sample_size_absolute,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_time": float(train_time),
        "infer_time_total": float(infer_time_total),
        "infer_time_per_sample": float(infer_time_per_sample),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance_top5": feature_importance_top5,
    }

    if save_result:
        save_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
        os.makedirs(save_dir, exist_ok=True)

        if sample_size_absolute is not None:
            save_filename = f"xrfm_{dataset_name}_{sample_size_absolute}.json"
        else:
            save_filename = f"xrfm_{dataset_name}.json"

        save_path = os.path.join(save_dir, save_filename)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        print(f"Saved to: {save_path}")
    else:
        print("Individual result saving skipped.")

    print("\n=== Result ===")
    print(f"Train time: {train_time:.4f} s")
    print(f"Inference total: {infer_time_total:.6f} s")
    print(f"Inference per sample: {infer_time_per_sample:.8f} s")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Feature importance top5: {feature_importance_top5}")

    return result_dict


if __name__ == "__main__":
    result = train_and_evaluate("diamonds", "regression")
    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))