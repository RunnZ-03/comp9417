import os
import json
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGRESSION_METRICS = ["rmse"]
CLASSIFICATION_METRICS = ["accuracy", "auc"]
TIME_METRICS = ["train_time", "infer_time_per_sample"]


def main():
    all_results = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json") and not filename.startswith("summary_"):
            file_path = os.path.join(RESULTS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                all_results.append(result)
            except Exception as e:
                pass
    if not all_results:
        return
    df_list = []
    for res in all_results:
        row = {
            "dataset": res["dataset"],
            "model": res["model"],
            "task_type": res["task_type"],
            "n_train": res["n_train"],
            "n_features": res["n_features"],
        }
        for metric, value in res["test_metrics"].items():
            row[f"test_{metric}"] = value
        for metric in TIME_METRICS:
            row[metric] = res[metric]
        df_list.append(row)
    df = pd.DataFrame(df_list)
    csv_save_path = os.path.join(RESULTS_DIR, "summary_all_results.csv")
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    reg_df = df[df["task_type"] == "regression"].copy()
    if not reg_df.empty:
        reg_table = reg_df.pivot(
            index="dataset", 
            columns="model", 
            values=["test_rmse", "train_time", "infer_time_per_sample"]
        ).round(6)
        tex_save_path = os.path.join(RESULTS_DIR, "summary_regression_table.tex")
        reg_table.to_latex(tex_save_path, caption="Regression Task Results", label="tab:regression_results")
    cls_df = df[df["task_type"] == "classification"].copy()
    if not cls_df.empty:
        cls_table = cls_df.pivot(
            index="dataset", 
            columns="model", 
            values=["test_accuracy", "test_auc", "train_time", "infer_time_per_sample"]
        ).round(6)
        tex_save_path = os.path.join(RESULTS_DIR, "summary_classification_table.tex")
        cls_table.to_latex(tex_save_path, caption="Classification Task Results", label="tab:classification_results")


if __name__ == "__main__":
    main()
