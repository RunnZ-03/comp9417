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
    return {"rmse": float(rmse)}


# =========================
# 安全提取 AGOP / feature importance
# =========================
def extract_feature_importance(model, features):
    """
    尝试多种方式提取 feature importance / AGOP diagonal。
    如果失败，返回错误信息，便于后续调试。
    """
    try:
        # 方式 1：如果模型直接有 get_feature_importance
        if hasattr(model, "get_feature_importance"):
            values = model.get_feature_importance()

            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            elif hasattr(values, "cpu"):
                values = values.cpu().numpy()
            elif hasattr(values, "tolist"):
                values = np.array(values.tolist())
            else:
                values = np.array(values)

            values = np.ravel(values)
            pairs = list(zip(features, values))
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
            return pairs[:5]

        # 方式 2：如果模型对象里暴露了 agop / feature_importance 之类属性
        candidate_attrs = [
            "feature_importance",
            "feature_importances_",
            "agop",
            "agop_diag",
            "diag_importance"
        ]

        for attr in candidate_attrs:
            if hasattr(model, attr):
                values = getattr(model, attr)

                if hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                elif hasattr(values, "cpu"):
                    values = values.cpu().numpy()
                elif hasattr(values, "tolist"):
                    values = np.array(values.tolist())
                else:
                    values = np.array(values)

                values = np.ravel(values)
                pairs = list(zip(features, values))
                pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
                return pairs[:5]

        return "Could not extract AGOP: no supported method/attribute found"

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

    # 可选：GPU 时做一点保守设置
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

                # 二分类取正类概率
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

    # AGOP / 重要性
    feature_importance_top5 = extract_feature_importance(model, features)

    # 封装结果
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
        "feature_importance_top5": feature_importance_top5
    }

    # 保存单个结果
    save_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"xrfm_{dataset_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print("=== Result ===")
    print(f"Train time: {train_time:.4f} s")
    print(f"Inference total: {infer_time_total:.6f} s")
    print(f"Inference per sample: {infer_time_per_sample:.8f} s")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics: {test_metrics}")
    print(f"Feature importance top5: {feature_importance_top5}")
    print(f"Saved to: {save_path}")

    return result_dict


if __name__ == "__main__":
    # 先跑一个测试
    result = train_and_evaluate("stroke", "classification")
    print("\nFinal result:")
    print(json.dumps(result, indent=4, ensure_ascii=False))