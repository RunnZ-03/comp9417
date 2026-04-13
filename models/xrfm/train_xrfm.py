import os
import sys
import json
import time
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

# =========================
# 路径设置
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset
from xrfm import xRFM

# =========================
# 随机种子
# =========================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 评估函数
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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"rmse": float(rmse)}


# =========================
# 工具函数
# =========================
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

    # 如果是矩阵，取对角线
    if values.ndim == 2:
        if values.shape[0] == values.shape[1]:
            values = np.diag(values)
        else:
            values = values.flatten()

    # 如果是多维，flatten
    if values.ndim > 1:
        values = values.flatten()

    # 如果只有一个值但feature很多，重复扩展
    if len(values) == 1 and len(features) > 1:
        values = np.repeat(values, len(features))

    # 长度不匹配时截断
    if len(values) != len(features):
        min_len = min(len(values), len(features))
        values = values[:min_len]
        features = features[:min_len]

    pairs = list(zip(features, values.tolist()))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return pairs[:5]


# =========================
# AGOP / Feature Importance 提取
# =========================
def extract_feature_importance(model, features):
    """
    当前 xRFM 版本的 AGOP 信息主要在:
    model.trees[0]["model"]
    优先尝试:
    1. diag
    2. M 的对角线
    3. agop_best_model
    """
    try:
        if hasattr(model, "trees") and isinstance(model.trees, list) and len(model.trees) > 0:
            tree0 = model.trees[0]

            if isinstance(tree0, dict) and "model" in tree0:
                inner_model = tree0["model"]

                # 1. 优先取 diag
                if hasattr(inner_model, "diag"):
                    diag_values = getattr(inner_model, "diag")
                    result = build_top5_from_values(diag_values, features)
                    if result is not None:
                        return result

                # 2. 再取 M 的对角线
                if hasattr(inner_model, "M"):
                    M = getattr(inner_model, "M")
                    result = build_top5_from_values(M, features)
                    if result is not None:
                        return result

                # 3. 再尝试 agop_best_model
                if hasattr(inner_model, "agop_best_model"):
                    agop_obj = getattr(inner_model, "agop_best_model")

                    if hasattr(agop_obj, "diag"):
                        result = build_top5_from_values(getattr(agop_obj, "diag"), features)
                        if result is not None:
                            return result

                    if hasattr(agop_obj, "M"):
                        result = build_top5_from_values(getattr(agop_obj, "M"), features)
                        if result is not None:
                            return result

        return "Could not extract AGOP from model.trees[0]['model']"

    except Exception as e:
        return f"Could not extract AGOP: {repr(e)}"


# =========================
# 核心训练函数
# =========================
def train_and_evaluate(dataset_name: str, task_type: str) -> Dict[str, Any]:
    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)

    print(f"\n--- 正在运行数据集: {dataset_name} ---")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"device: {DEVICE}")

    # 初始化模型
    model = xRFM(use_diag=True, device=DEVICE)
    print(model)

    # GPU 保守优化
    try:
        if DEVICE == "cuda" and hasattr(model, "rfm_params"):
            if "fit" in model.rfm_params:
                model.rfm_params["fit"]["n_iter"] = 3
                model.rfm_params["fit"]["M_batch_size"] = 128
                print("GPU 优化：n_iter=3, M_batch_size=128")
    except Exception as e:
        print(f"跳过 rfm_params 调整: {repr(e)}")

    # 训练
    start_train = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_train

    # 验证集预测
    val_pred = model.predict(X_val)

    # 测试集预测
    start_infer = time.time()
    test_pred = model.predict(X_test)
    infer_time_total = time.time() - start_infer
    infer_time_per_sample = infer_time_total / max(len(X_test), 1)

    # 指标
    if task_type == "classification":
        val_prob = None
        test_prob = None

        if hasattr(model, "predict_proba"):
            try:
                val_prob_raw = model.predict_proba(X_val)
                test_prob_raw = model.predict_proba(X_test)

                if len(np.shape(val_prob_raw)) == 2 and np.shape(val_prob_raw)[1] >= 2:
                    val_prob = val_prob_raw[:, 1]
                    test_prob = test_prob_raw[:, 1]
            except Exception as e:
                print(f"predict_proba 获取失败: {repr(e)}")

        val_metrics = evaluate_classification(y_val, val_pred, val_prob)
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)

    else:
        val_metrics = evaluate_regression(y_val, val_pred)
        test_metrics = evaluate_regression(y_test, test_pred)

    # AGOP / feature importance
    feature_importance_top5 = extract_feature_importance(model, features)

    # 结果封装
    result_dict = {
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

    # 保存单个结果
    save_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"xrfm_{dataset_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print("\n=== Result ===")
    print(f"Train time: {train_time:.4f} s")
    print(f"Inference total: {infer_time_total:.6f} s")
    print(f"Inference per sample: {infer_time_per_sample:.8f} s")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Feature importance top5: {feature_importance_top5}")
    print(f"Saved to: {save_path}")

    return result_dict


if __name__ == "__main__":
    # 默认单独测试一个
    result = train_and_evaluate("diamonds", "regression")
    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))