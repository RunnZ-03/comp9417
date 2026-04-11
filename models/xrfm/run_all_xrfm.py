import os
import sys
import json
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_parent = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)

from comp9417.models.xrfm.train_xrfm import train_and_evaluate

def run_all():
    # 严格对应 data_loader 支持的别名
    # 组合：2回归 + 3分类，包含 n>10000 和 d>50 
    tasks = [
        ("diamonds", "regression"),          # n=53k, 回归
        ("superconductivity", "regression"), # d=82, 回归
        ("shoppers", "classification"),      # 混合特征, 分类
        ("stroke", "classification"),        # 混合特征, 分类
        ("hr_attrition", "classification")   # 分类
    ]
    
    summary = []
    for name, t_type in tasks:
        try:
            print(f"\n正在处理任务: {name}...")
            res = train_and_evaluate(name, t_type)
            summary.append(res)
            # 及时释放显存
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"任务 {name} 失败: {e}")

    # 保存最终汇总表供组员 E 绘图
    with open(f"{project_parent}/comp9417/results/summary_all_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\n所有任务已完成！请检查 results 文件夹。")

if __name__ == "__main__":
    run_all()