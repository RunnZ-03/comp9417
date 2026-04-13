import os
import sys
import json
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.models.xrfm.train_xrfm import train_and_evaluate


def run_all():
    tasks = [
        ("diamonds", "regression"),
        ("superconductivity", "regression"),
        ("shoppers", "classification"),
        ("stroke", "classification"),
        ("hr_attrition", "classification"),
    ]

    summary = []

    for dataset_name, task_type in tasks:
        try:
            print("\n==============================")
            print(f"开始处理: {dataset_name} ({task_type})")
            print("==============================")

            result = train_and_evaluate(dataset_name, task_type)
            summary.append(result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"任务 {dataset_name} 失败: {repr(e)}")
            summary.append({
                "dataset": dataset_name,
                "task_type": task_type,
                "error": repr(e)
            })

    results_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary_all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n所有任务已完成。")
    print(f"汇总结果保存在: {summary_path}")


if __name__ == "__main__":
    run_all()