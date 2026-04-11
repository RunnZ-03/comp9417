import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import sys
import os
import torch
import json
from datetime import datetime

# 固定随机种子以保证可复现性（COMP9417 实验规范） [cite: 178]
np.random.seed(42)
torch.manual_seed(42)

# 显存清理
if torch.cuda.is_available():
    torch.cuda.empty_cache()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_parent = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

from comp9417.data.data_loader import get_dataset
from xrfm import xRFM

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_classification(y_true, y_pred, y_prob=None):
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if y_prob is not None:
        try:
            # 作业要求报告 AUC-ROC [cite: 180]
            result["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            result["auc"] = None
    return result

def evaluate_regression(y_true, y_pred):
    # 作业要求报告 RMSE [cite: 180]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"rmse": float(rmse)}

def train_and_evaluate(dataset_name, task_type):
    # 加载数据集
    X_train, X_val, X_test, y_train, y_val, y_test, features = get_dataset(dataset_name)
    print(f"--- 正在运行数据集: {dataset_name} ---")
    
    # 初始化模型，使用对角线模式以匹配表格数据结构 [cite: 389, 959]
    model = xRFM(use_diag=True, device=device)

    # 针对 T4 显存优化：增加迭代次数以提升特征学习效果 [cite: 342]
    if device == "cuda":
        model.rfm_params['fit']['n_iter'] = 3 # 提升至 3 次迭代以利用 AGOP 特征学习 [cite: 342, 840]
        model.rfm_params['fit']['M_batch_size'] = 128
        print(f"显存优化：M_batch_size=128, n_iter=3")

    # 训练
    start_train = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_train

    # 推理
    start_infer = time.time()
    test_pred = model.predict(X_test)
    infer_time = time.time() - start_infer
    
    # 提取特征重要性 (AGOP 对角线) [cite: 305, 703]
    # 这是报告中“讨论”部分的关键数据 [cite: 92, 191]
    feature_importance = {}
    try:
        # 假设 leaf RFM 的权重可以通过此接口获取
        agop_diag = model.get_feature_importance() 
        feature_importance = dict(zip(features, agop_diag.tolist()))
    except:
        feature_importance = "Could not extract AGOP"

    # 指标计算
    if task_type == "classification":
        test_prob = None
        if hasattr(model, "predict_proba"):
            test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_classification(y_test, test_pred, test_prob)
    else:
        test_metrics = evaluate_regression(y_test, test_pred)

    # 结果封装
    result_dict = {
        "dataset": dataset_name,
        "task_type": task_type,
        "metrics": test_metrics,
        "train_time": float(train_time),
        "infer_time_per_sample": float(infer_time / len(X_test)),
        "feature_importance_top5": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5] if isinstance(feature_importance, dict) else feature_importance
    }
    
    # 自动保存
    save_path = os.path.join(project_parent, "comp9417/results")
    if not os.path.exists(save_path): os.makedirs(save_path)
    with open(f"{save_path}/xrfm_{dataset_name}.json", "w") as f:
        json.dump(result_dict, f, indent=4)
    
    return result_dict