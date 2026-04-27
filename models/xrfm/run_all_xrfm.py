import os
import sys
import json
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.models.xrfm.train_xrfm import train_and_evaluate


def run_standard_experiments():
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
            print(f"Running: {dataset_name} ({task_type})")
            print("==============================")

            result = train_and_evaluate(dataset_name, task_type)
            summary.append(result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Task failed: {dataset_name}, error: {repr(e)}")
            summary.append({
                "dataset": dataset_name,
                "task_type": task_type,
                "error": repr(e)
            })

    return summary


def run_diamonds_scaling_experiment():
    sample_sizes = [1000, 5000, 10000, 30000]
    scaling_results = []

    for sample_size in sample_sizes:
        try:
            print("\n==============================")
            print(f"Running Diamonds scaling experiment: {sample_size}")
            print("==============================")

            result = train_and_evaluate(
                dataset_name="diamonds",
                task_type="regression",
                sample_size_absolute=sample_size
            )

            scaling_results.append({
                "dataset": "diamonds",
                "sample_size": sample_size,
                "n_train": result["n_train"],
                "rmse": result["test_metrics"]["rmse"],
                "train_time": result["train_time"],
                "infer_time_total": result["infer_time_total"],
                "infer_time_per_sample": result["infer_time_per_sample"]
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Scaling task failed: sample_size={sample_size}, error: {repr(e)}")
            scaling_results.append({
                "dataset": "diamonds",
                "sample_size": sample_size,
                "error": repr(e)
            })

    return scaling_results


def save_json(data, filename):
    results_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved to: {save_path}")


def run_all():
    summary = run_standard_experiments()
    save_json(summary, "summary_all_results.json")

    scaling_results = run_diamonds_scaling_experiment()
    save_json(scaling_results, "xrfm_scaling_results.json")

    print("\nAll xRFM experiments completed.")


if __name__ == "__main__":
    run_all()