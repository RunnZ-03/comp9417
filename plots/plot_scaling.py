import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def load_scaling_data(dataset_name="diamonds"):
    models = {
        'LightGBM': 'results/lightgbm_scaling_results.json',
        'XGBoost': 'results/xgboost_scaling_results.json',
        'MLP': 'results/mlp_scaling_results.json',
        'xRFM': 'results/xrfm_scaling_results.json'
    }

    all_data = []
    for model_name, path in models.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                for row in data:
                    if row['dataset'] == dataset_name:
                        row['Model'] = model_name
                        all_data.append(row)
        else:
            print(f"Warning: {path} not found. Skipping {model_name}.")

    return pd.DataFrame(all_data)


def plot_scaling_results():
    df = load_scaling_data(dataset_name="diamonds")
    if df.empty:
        print("No data found for plotting.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    sns.lineplot(data=df, x='n_train', y='rmse', hue='Model', marker='o', ax=axes[0], linewidth=2)
    axes[0].set_title('Test RMSE vs. Training Size (Diamonds Dataset)', fontsize=14, pad=10)
    axes[0].set_xlabel('Number of Training Samples (n)', fontsize=12)
    axes[0].set_ylabel('Test RMSE (Lower is better)', fontsize=12)


    sns.lineplot(data=df, x='n_train', y='train_time', hue='Model', marker='s', ax=axes[1], linewidth=2)
    axes[1].set_title('Training Time vs. Training Size', fontsize=14, pad=10)
    axes[1].set_xlabel('Number of Training Samples (n)', fontsize=12)
    axes[1].set_ylabel('Training Time (Seconds)', fontsize=12)
    axes[1].set_yscale('log')  

    plt.tight_layout()


    os.makedirs('report/figures', exist_ok=True)
    save_path = 'report/figures/scaling_analysis.pdf'
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Plot saved successfully to {save_path}")


if __name__ == "__main__":
    plot_scaling_results()