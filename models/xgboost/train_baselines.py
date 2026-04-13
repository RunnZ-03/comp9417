import os
import sys
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# =========================
# Path setup (mirrors xrfm)
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset

np.random.seed(42)

RESULTS_DIR = os.path.join(PROJECT_PARENT, "comp9417", "results")


# =========================
# Evaluation helpers
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


def get_top5_features(importances, features) -> Any:
    try:
        importances = np.array(importances)
        if len(importances) != len(features):
            return None
        pairs = sorted(zip(features, importances.tolist()), key=lambda x: x[1], reverse=True)
        return [list(p) for p in pairs[:5]]
    except Exception as e:
        return f"Could not extract feature importance: {repr(e)}"


# =========================
# XGBoost
# =========================
def _train_xgboost(X_train, y_train, X_val, y_val, X_test, task_type):
    import xgboost as xgb

    common = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    if task_type == "classification":
        model = xgb.XGBClassifier(
            **common,
            eval_metric="logloss",
            early_stopping_rounds=20,
        )
    else:
        model = xgb.XGBRegressor(
            **common,
            eval_metric="rmse",
            early_stopping_rounds=20,
        )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - t0

    val_pred = model.predict(X_val)

    t1 = time.time()
    test_pred = model.predict(X_test)
    infer_time = time.time() - t1

    return model, val_pred, test_pred, train_time, infer_time


# =========================
# LightGBM
# =========================
def _train_lightgbm(X_train, y_train, X_val, y_val, X_test, task_type):
    import lightgbm as lgb

    common = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    if task_type == "classification":
        model = lgb.LGBMClassifier(**common)
    else:
        model = lgb.LGBMRegressor(**common)

    t0 = time.time()
    try:
        # LightGBM >= 3.3 callback API
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    except TypeError:
        # Older LightGBM fallback
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False,
        )
    train_time = time.time() - t0

    val_pred = model.predict(X_val)

    t1 = time.time()
    test_pred = model.predict(X_test)
    infer_time = time.time() - t1

    return model, val_pred, test_pred, train_time, infer_time


# =========================
# Unified training entry point
# =========================
def train_and_evaluate(model_name: str, dataset_name: str, task_type: str) -> Dict[str, Any]:
    """Train one model on one dataset and return / save a result dict.

    Args:
        model_name:   "xgboost" | "lightgbm"
        dataset_name: key in datasets_config (e.g. "diamonds")
        task_type:    "regression" | "classification"
    """
    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)

    # Ensure labels are integer-encoded (handles ArrowStringArray / object dtypes)
    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)
    if task_type == "classification" and y_train.dtype.kind not in ("i", "u", "b"):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val   = le.transform(y_val)
        y_test  = le.transform(y_test)

    print(f"\n--- {model_name.upper()} | {dataset_name} | {task_type} ---")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    dispatch = {
        "xgboost": _train_xgboost,
        "lightgbm": _train_lightgbm,
    }
    if model_name not in dispatch:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(dispatch)}")

    model, val_pred, test_pred, train_time, infer_time_total = dispatch[model_name](
        X_train, y_train, X_val, y_val, X_test, task_type
    )

    infer_time_per_sample = infer_time_total / max(len(X_test), 1)

    # --- Metrics ---
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
        else:
            val_prob = test_prob = None
        val_metrics = evaluate_classification(y_val, val_pred, val_prob)
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)
    else:
        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

    # --- Feature importance ---
    try:
        feature_importance_top5 = get_top5_features(model.feature_importances_, features)
    except Exception as e:
        feature_importance_top5 = f"N/A: {repr(e)}"

    # --- Package result ---
    result = {
        "model": model_name,
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
        "feature_importance_top5": feature_importance_top5,
    }

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Train time:  {train_time:.4f} s")
    print(f"Infer total: {infer_time_total:.6f} s  ({infer_time_per_sample:.2e} s/sample)")
    print(f"Val  metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Top-5 features: {feature_importance_top5}")
    print(f"Saved → {save_path}")

    return result


if __name__ == "__main__":
    result = train_and_evaluate("xgboost", "diamonds", "regression")
    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))
