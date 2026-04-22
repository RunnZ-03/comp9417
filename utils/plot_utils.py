import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "figure.figsize": (10, 6),
    "savefig.bbox": "tight",
    "savefig.format": "png"
})

COLORS = {
    "agop": "#1f77b4",
    "pca": "#ff7f0e",
    "mutual_info": "#2ca02c",
    "permutation_importance": "#d62728",
    "xrfm": "#1f77b4",
    "xgboost": "#ff7f0e",
    "lightgbm": "#2ca02c",
    "mlp": "#d62728",
    "random_forest": "#9467bd"
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_feature_importance_comparison(
    agop_importance: Dict[str, float],
    baseline_importance: Dict[str, Dict[str, float]],
    dataset_name: str,
    top_k: int = 10,
    save: bool = True
):
    all_features = list(agop_importance.keys())
    df = pd.DataFrame(index=all_features)
    df["agop"] = pd.Series(agop_importance).abs()
    for name, imp in baseline_importance.items():
        df[name] = pd.Series(imp).abs()
    df = df.sort_values("agop", ascending=False).head(top_k)
    df = df / df.max()
    x = np.arange(len(df.index))
    width = 0.2
    n_methods = len(df.columns)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(df.columns):
        offset = width * (i - (n_methods-1)/2)
        ax.bar(x + offset, df[col], width, label=col.upper(), color=COLORS.get(col, f"C{i}"))
    ax.set_title(f"Feature Importance Comparison - {dataset_name.capitalize()} Dataset", pad=15)
    ax.set_ylabel("Normalized Importance (0-1)")
    ax.set_xlabel("Feature Name")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    if save:
        save_path = os.path.join(PLOTS_DIR, f"{dataset_name}_feature_importance_comparison.png")
        plt.savefig(save_path)
    plt.show()
    return fig, ax


def plot_scalability_curve(
    sample_sizes: List[int],
    performance_results: Dict[str, List[float]],
    time_results: Dict[str, List[float]],
    dataset_name: str,
    task_type: str,
    performance_metric: str,
    save: bool = True
):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel(f"Test {performance_metric}", color="black")
    for model_name, values in performance_results.items():
        ax1.plot(
            sample_sizes, values, 
            marker="o", linestyle="-", linewidth=2,
            label=f"{model_name} ({performance_metric})",
            color=COLORS.get(model_name.lower(), None)
        )
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(axis="both", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Time (seconds)", color="gray")
    for model_name, values in time_results.items():
        ax2.plot(
            sample_sizes, values,
            marker="s", linestyle="--", linewidth=1.5,
            label=f"{model_name} (Train Time)",
            color=COLORS.get(model_name.lower(), None),
            alpha=0.7
        )
    ax2.tick_params(axis="y", labelcolor="gray")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    metric_direction = "Lower is Better" if performance_metric == "RMSE" else "Higher is Better"
    ax1.set_title(
        f"Model Scalability - {dataset_name.capitalize()} {task_type.capitalize()} Task\n({performance_metric}: {metric_direction})",
        pad=15
    )
    if save:
        save_path = os.path.join(PLOTS_DIR, f"{dataset_name}_scalability_curve.png")
        plt.savefig(save_path)
    plt.show()
    return fig, ax1, ax2
