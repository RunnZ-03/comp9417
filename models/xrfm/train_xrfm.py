import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import sys
import os
import torch
import json
from datetime import datetime
import torch
# 1. 释放 PyTorch 缓存
torch.cuda.empty_cache()
# 强制清理显存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==========================================
# 1. 自动路径修复逻辑 (解决 ModuleNotFoundError)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_parent = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

# 确保导入顺序
from comp9417.data.data_loader import get_dataset
from xrfm import xRFM

# ==========================================
# 2. 评估函数定义
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

def evaluate_classification(y_true, y_pred, y_prob=None):
    result = {"accuracy": accuracy_score(y_true, y_pred)}
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

# ==========================================
# 3. 核心训练与保存逻辑
# ==========================================
def train_and_evaluate(dataset_name, task_type):
    # 加载数据集
    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)

    print(f"--- Running dataset: {dataset_name} ---")
    
    # 根据硬件调整样本量 (5000 样本在 T4 上是安全的)

    # ==========================================
    # 初始化 xRFM 模型 (原地修改策略)
    # ==========================================
    # 1. 先用标准方式初始化，让库自动填充所有默认参数 (kernel, reg, etc.)
    model = xRFM(use_diag=True, device=device)

    # 2. 只有在 GPU 环境下才需要手动压制批处理量，防止 5.92GB 溢出报错
    if device == "cuda":
        # 精准修改 fit 字典中的参数，避免覆盖导致的 KeyError
        model.rfm_params['fit']['n_iter'] = 1
        model.rfm_params['fit']['M_batch_size'] = 128
        print(f"已手动优化 M_batch_size 为: {model.rfm_params['fit']['M_batch_size']}")

    # 训练模型
    start_train = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_train

    # 推理
    start_infer = time.time()
    test_pred = model.predict(X_test)
    inference_time = time.time() - start_infer
    inference_time_per_sample = inference_time / len(X_test)

    # 计算指标
    if task_type == "classification":
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)
    else:
        test_metrics = evaluate_regression(y_test, test_pred)

    # 打印结果
    print(f"Train time: {train_time:.4f} s")
    print(f"Inference time per sample: {inference_time_per_sample:.8f} s")
    print(f"Test metrics: {test_metrics}")

    # 整合结果字典
    result_dict = {
        "dataset": dataset_name,
        "task_type": task_type,
        "train_time": train_time,
        "inference_time_per_sample": inference_time_per_sample,
        "test_metrics": test_metrics,
        "model_params": {"use_diag": True, "n_train": len(X_train), "M_batch_size": 128}
    }
    
    # 保存结果
    save_path = os.path.join(project_parent, "comp9417/results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with open(f"{save_path}/xrfm_{dataset_name}_{timestamp}.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    
    return result_dict

if __name__ == "__main__":
    # 执行回归任务
    final_result = train_and_evaluate("diamonds", "regression")