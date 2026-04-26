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
    
    # 1. 生成Excel能打开的CSV文件（和之前完全一致）
    df = pd.DataFrame(all_results)
    csv_save_path = os.path.join(RESULTS_DIR, "summary_all_results.csv")
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV结果表已保存: {csv_save_path}")

    # 2. 生成LaTeX表格（加了异常保护，不会崩溃）
    # 2.1 回归任务LaTeX表格
    reg_df = df[df["task_type"] == "regression"].copy()
    if not reg_df.empty:
        try:
            reg_table = reg_df.pivot(
                index="dataset",
                columns="model",
                values=["test_rmse", "train_time", "infer_time_per_sample"]
            ).round(6)
            tex_reg_path = os.path.join(RESULTS_DIR, "summary_regression_table.tex")
            reg_table.to_latex(
                tex_reg_path,
                caption="Regression Task Performance & Time Comparison",
                label="tab:regression_results",
                position="htbp",
                column_format="l" + "c"*len(reg_table.columns)
            )
            print(f"✅ 回归任务LaTeX表格已保存: {tex_reg_path}")
        except Exception as e:
            print(f"⚠️  回归LaTeX表格生成失败: {str(e)}")
            print("💡 请先运行 pip3 install jinja2 安装依赖")

    # 2.2 分类任务LaTeX表格
    cls_df = df[df["task_type"] == "classification"].copy()
    if not cls_df.empty:
        try:
            cls_table = cls_df.pivot(
                index="dataset",
                columns="model",
                values=["test_accuracy", "test_auc", "train_time", "infer_time_per_sample"]
            ).round(6)
            tex_cls_path = os.path.join(RESULTS_DIR, "summary_classification_table.tex")
            cls_table.to_latex(
                tex_cls_path,
                caption="Classification Task Performance & Time Comparison",
                label="tab:classification_results",
                position="htbp",
                column_format="l" + "c"*len(cls_table.columns)
            )
            print(f"✅ 分类任务LaTeX表格已保存: {tex_cls_path}")
        except Exception as e:
            print(f"⚠️  分类LaTeX表格生成失败: {str(e)}")
            print("💡 请先运行 pip3 install jinja2 安装依赖")
    
    # 最终汇总提示
    print(f"\n=== 运行完成汇总 ===")
    print(f"成功处理 {len(all_results)} 个有效结果文件")
    if skipped_files:
        print(f"跳过 {len(skipped_files)} 个无效文件")


if __name__ == "__main__":
    main()
