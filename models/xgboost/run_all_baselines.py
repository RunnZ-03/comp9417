import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.models.xgboost.train_baselines import train_and_evaluate

TASKS = [
    ("diamonds",          "regression"),
    ("superconductivity", "regression"),
    ("shoppers",          "classification"),
    ("stroke",            "classification"),
    ("hr_attrition",      "classification"),
]

MODELS = ["xgboost", "lightgbm"]


def run_all():
    summary = []

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name.upper()}")
        print(f"{'='*60}")

        for dataset_name, task_type in TASKS:
            try:
                result = train_and_evaluate(model_name, dataset_name, task_type)
                summary.append(result)
            except Exception as e:
                print(f"  [ERROR] {model_name} / {dataset_name}: {repr(e)}")
                summary.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "error": repr(e),
                })

    # Save combined summary
    results_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary_baselines.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nAll results saved → {summary_path}")

    # Print comparison table
    _print_comparison(summary)


def _print_comparison(summary):
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)

    regression_rows = [(r["dataset"], r["model"], r["test_metrics"].get("rmse"))
                       for r in summary if r.get("task_type") == "regression" and "test_metrics" in r]

    clf_rows = [(r["dataset"], r["model"],
                 r["test_metrics"].get("accuracy"), r["test_metrics"].get("auc"))
                for r in summary if r.get("task_type") == "classification" and "test_metrics" in r]

    if regression_rows:
        print("\nRegression — Test RMSE (lower is better)")
        print(f"{'Dataset':<25} {'Model':<16} {'RMSE':>12}")
        print("-" * 55)
        for dataset, model, rmse in sorted(regression_rows, key=lambda x: (x[0], x[1])):
            rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
            print(f"{dataset:<25} {model:<16} {rmse_str:>12}")

    if clf_rows:
        print("\nClassification — Test Accuracy & AUC (higher is better)")
        print(f"{'Dataset':<20} {'Model':<16} {'Accuracy':>10} {'AUC':>10}")
        print("-" * 60)
        for dataset, model, acc, auc in sorted(clf_rows, key=lambda x: (x[0], x[1])):
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"{dataset:<20} {model:<16} {acc_str:>10} {auc_str:>10}")

    # Include xRFM results if the summary file exists
    xrfm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results", "summary_all_results.json"
    )
    if os.path.exists(xrfm_path):
        with open(xrfm_path, encoding="utf-8") as f:
            xrfm_data = json.load(f)

        print("\n--- xRFM reference (from teammate results) ---")
        print(f"{'Dataset':<25} {'Task':<16} {'RMSE / Accuracy':<18} {'AUC':>8}")
        print("-" * 70)
        for r in xrfm_data:
            dataset = r.get("dataset", "")
            task = r.get("task_type", "")
            tm = r.get("test_metrics", {})
            if task == "regression":
                metric_str = f"RMSE={tm.get('rmse', 'N/A'):.4f}" if tm.get("rmse") is not None else "N/A"
                auc_str = "—"
            else:
                acc = tm.get("accuracy")
                auc = tm.get("auc")
                metric_str = f"Acc={acc:.4f}" if acc is not None else "N/A"
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"{dataset:<25} {task:<16} {metric_str:<18} {auc_str:>8}")


if __name__ == "__main__":
    run_all()
