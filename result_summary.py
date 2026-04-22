import os
import json
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    
    df = pd.DataFrame(all_results)
    csv_save_path = os.path.join(RESULTS_DIR, "summary_all_results.csv")
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"✅ Full results saved to: {csv_save_path}")
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(all_results)} valid result files.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} invalid files.")


if __name__ == "__main__":
    main()
