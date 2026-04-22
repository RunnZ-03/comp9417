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
    skipped_files = []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json") and not filename.startswith("summary_"):
            file_path = os.path.join(RESULTS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                
                required_fields = ["model", "dataset", "task_type", "test_metrics", "train_time", "infer_time_per_sample"]
                if not all(field in res for field in required_fields):
                    skipped_files.append(filename)
                    continue
                
                row = {
                    "dataset": res["dataset"],
                    "model": res["model"],
                    "task_type": res["task_type"],
                    "n_train": res.get("n_train", None),
                    "n_features": res.get("n_features", None),
                }
                
                for metric, value in res["test_metrics"].items():
                    row[f"test_{metric}"] = value
                
                for metric in TIME_METRICS:
                    row[metric] = res[metric]
                
                all_results.append(row)
            except Exception as e:
                skipped_files.append(filename)
    
    if not all_results:
        print("No valid result files found.")
        if skipped_files:
            print(f"Skipped {len(skipped_files)} invalid files.")
        return
    
    df = pd.DataFrame(df_list)
    csv_save_path = os.path.join(RESULTS_DIR, "summary_all_results.csv")
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"✅ Full results saved to: {csv_save_path}")
    
    reg_df = df[df["task_type"] == "regression"].copy()
    if not reg_df.empty:
        reg_table = reg_df.pivot(
            index="dataset", 
            columns="model", 
            values=["test_rmse", "train_time", "infer_time_per_sample"]
        ).round(6)
        tex_save_path = os.path.join(RESULTS_DIR, "summary_regression_table.tex")
        reg_table.to_latex(tex_save_path, caption="Regression Task Results", label="tab:regression_results")
        print(f"✅ Regression LaTeX table saved.")
    
    cls_df = df[df["task_type"] == "classification"].copy()
    if not cls_df.empty:
        cls_table = cls_df.pivot(
            index="dataset", 
            columns="model", 
            values=["test_accuracy", "test_auc", "train_time", "infer_time_per_sample"]
        ).round(6)
        tex_save_path = os.path.join(RESULTS_DIR, "summary_classification_table.tex")
        cls_table.to_latex(tex_save_path, caption="Classification Task Results", label="tab:classification_results")
        print(f"✅ Classification LaTeX table saved.")
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(all_results)} valid result files.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} invalid files: {skipped_files}")


if __name__ == "__main__":
    main()
