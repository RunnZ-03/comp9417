import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)

from comp9417.data.data_loader import get_dataset
from xrfm import xRFM


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_numpy(x):
    if x is None:
        return None

    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    if hasattr(x, "cpu"):
        return x.cpu().numpy()

    if hasattr(x, "values"):
        return x.values

    if hasattr(x, "tolist"):
        return np.array(x.tolist())

    return np.array(x)


def normalise_importance(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.array(values, dtype=float))
    max_value = np.max(values)

    if max_value == 0 or np.isnan(max_value):
        return values

    return values / max_value


def values_to_feature_dict(values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    values = normalise_importance(values)
    return {
        str(feature): float(value)
        for feature, value in zip(feature_names, values)
    }


def get_top_features(importance_dict: Dict[str, float], top_k: int = 10) -> List[Tuple[str, float]]:
    return sorted(
        importance_dict.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:top_k]


def convert_agop_values_to_vector(values, n_features: int) -> Optional[np.ndarray]:
    values = to_numpy(values)

    if values is None:
        return None

    values = np.array(values)

    if values.ndim == 0:
        return None

    if values.ndim == 2:
        if values.shape[0] == values.shape[1]:
            values = np.diag(values)
        else:
            values = values.reshape(-1)

    values = values.reshape(-1)

    if values.size != n_features:
        return None

    return values


def extract_agop_importance(model, feature_names: List[str]) -> Dict[str, float]:
    n_features = len(feature_names)

    try:
        if hasattr(model, "trees") and isinstance(model.trees, list) and len(model.trees) > 0:
            tree0 = model.trees[0]

            if isinstance(tree0, dict) and "model" in tree0:
                inner_model = tree0["model"]

                candidates = []

                if hasattr(inner_model, "M"):
                    candidates.append(getattr(inner_model, "M"))

                if hasattr(inner_model, "diag"):
                    candidates.append(getattr(inner_model, "diag"))

                if hasattr(inner_model, "agop_best_model"):
                    agop_obj = getattr(inner_model, "agop_best_model")

                    if hasattr(agop_obj, "M"):
                        candidates.append(getattr(agop_obj, "M"))

                    if hasattr(agop_obj, "diag"):
                        candidates.append(getattr(agop_obj, "diag"))

                for candidate in candidates:
                    vector = convert_agop_values_to_vector(candidate, n_features)

                    if vector is not None:
                        return values_to_feature_dict(vector, feature_names)

    except Exception as e:
        print(f"Could not extract AGOP importance: {repr(e)}")

    return {}


def train_xrfm_model(X_train, y_train, X_val, y_val):
    model = xRFM(use_diag=True, device=DEVICE)

    try:
        if DEVICE == "cuda" and hasattr(model, "rfm_params"):
            if "fit" in model.rfm_params:
                model.rfm_params["fit"]["n_iter"] = 3
                model.rfm_params["fit"]["M_batch_size"] = 128
    except Exception as e:
        print(f"Skipped GPU parameter adjustment: {repr(e)}")

    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time

    return model, train_time


def evaluate_model(model, X_test, y_test, task_type: str) -> Dict[str, float]:
    y_test = to_numpy(y_test).reshape(-1)
    y_pred = to_numpy(model.predict(X_test)).reshape(-1)

    if task_type == "regression":
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {"rmse": float(rmse)}

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred))
    }

    y_prob = None

    if hasattr(model, "predict_proba"):
        try:
            y_prob_raw = to_numpy(model.predict_proba(X_test))

            if y_prob_raw is not None:
                y_prob_raw = np.array(y_prob_raw)

                if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] >= 2:
                    y_prob = y_prob_raw[:, 1]
                elif y_prob_raw.ndim == 1:
                    y_prob = y_prob_raw

        except Exception as e:
            print(f"Could not get predict_proba: {repr(e)}")

    if y_prob is not None:
        try:
            result["auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            result["auc"] = None
    else:
        result["auc"] = None

    return result


def calculate_pca_importance(X_train, feature_names: List[str]) -> Dict[str, float]:
    pca = PCA(n_components=1, random_state=SEED)
    pca.fit(X_train)

    loadings = np.abs(pca.components_[0])

    return values_to_feature_dict(loadings, feature_names)


def calculate_mutual_info_importance(
    X_train,
    y_train,
    feature_names: List[str],
    task_type: str
) -> Dict[str, float]:

    y_train = to_numpy(y_train).reshape(-1)

    if task_type == "regression":
        scores = mutual_info_regression(
            X_train,
            y_train,
            random_state=SEED
        )
    else:
        scores = mutual_info_classif(
            X_train,
            y_train,
            random_state=SEED
        )

    return values_to_feature_dict(scores, feature_names)


def sample_for_permutation(X, y, max_samples: int = 2000):
    X = to_numpy(X)
    y = to_numpy(y).reshape(-1)

    if len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=max_samples, replace=False)

    return X[idx], y[idx]


def calculate_permutation_importance_scores(
    model,
    X_val,
    y_val,
    feature_names: List[str],
    task_type: str,
    max_samples: int = 2000,
    n_repeats: int = 5
) -> Dict[str, float]:

    X_perm, y_perm = sample_for_permutation(
        X_val,
        y_val,
        max_samples=max_samples
    )

    if task_type == "regression":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = "accuracy"

    result = permutation_importance(
        model,
        X_perm,
        y_perm,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=1
    )

    return values_to_feature_dict(result.importances_mean, feature_names)


def latex_escape_feature_name(feature: str) -> str:
    return feature.replace("_", r"\_")


def save_latex_table(
    top_features_by_method: Dict[str, List[Tuple[str, float]]],
    dataset_name: str,
    save_path: str
):
    method_labels = {
        "agop": "AGOP diagonal",
        "pca": "PCA loading",
        "mutual_info": "Mutual information",
        "permutation_importance": "Permutation importance"
    }

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(rf"  \caption{{Feature-importance comparison on the {dataset_name.capitalize()} dataset.}}")
    lines.append(rf"  \label{{tab:{dataset_name}-feature-importance}}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & Top ranked features \\")
    lines.append(r"    \midrule")

    for method, features in top_features_by_method.items():
        label = method_labels.get(method, method)
        feature_text = ", ".join(
            [rf"\texttt{{{latex_escape_feature_name(name)}}}" for name, _ in features[:5]]
        )
        lines.append(rf"    {label} & {feature_text} \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_importance_comparison(
    importance_by_method: Dict[str, Dict[str, float]],
    dataset_name: str,
    save_path: str,
    top_k: int = 10
):
    agop_importance = importance_by_method.get("agop", {})

    if agop_importance:
        selected_features = [name for name, _ in get_top_features(agop_importance, top_k)]
    else:
        all_scores = {}
        for importance_dict in importance_by_method.values():
            for feature, value in importance_dict.items():
                all_scores[feature] = max(all_scores.get(feature, 0.0), abs(value))
        selected_features = [
            name for name, _ in get_top_features(all_scores, top_k)
        ]

    methods = list(importance_by_method.keys())
    x = np.arange(len(selected_features))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(14, 6))

    for i, method in enumerate(methods):
        values = [
            importance_by_method[method].get(feature, 0.0)
            for feature in selected_features
        ]

        offset = (i - (len(methods) - 1) / 2) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=method
        )

    plt.xticks(
        x,
        [feature.replace("num__", "").replace("cat__", "") for feature in selected_features],
        rotation=45,
        ha="right"
    )
    plt.ylabel("Normalised importance")
    plt.xlabel("Feature")
    plt.title(f"Feature Importance Comparison on {dataset_name.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_interpretability_experiment(
    dataset_name: str,
    task_type: str,
    top_k: int = 10,
    max_perm_samples: int = 2000,
    n_repeats: int = 5
):
    print("\n==============================")
    print(f"Running interpretability experiment: {dataset_name} ({task_type})")
    print("==============================")
    print(f"Device: {DEVICE}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = get_dataset(dataset_name)

    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")

    model, train_time = train_xrfm_model(X_train, y_train, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test, task_type)

    agop_importance = extract_agop_importance(model, feature_names)
    pca_importance = calculate_pca_importance(X_train, feature_names)
    mutual_info_importance = calculate_mutual_info_importance(
        X_train,
        y_train,
        feature_names,
        task_type
    )
    permutation_scores = calculate_permutation_importance_scores(
        model,
        X_val,
        y_val,
        feature_names,
        task_type,
        max_samples=max_perm_samples,
        n_repeats=n_repeats
    )

    importance_by_method = {
        "agop": agop_importance,
        "pca": pca_importance,
        "mutual_info": mutual_info_importance,
        "permutation_importance": permutation_scores
    }

    top_features_by_method = {
        method: get_top_features(scores, top_k=top_k)
        for method, scores in importance_by_method.items()
    }

    results_dir = os.path.join(PROJECT_PARENT, "comp9417", "results")
    figures_dir = os.path.join(PROJECT_PARENT, "comp9417", "report", "figures")
    tables_dir = os.path.join(PROJECT_PARENT, "comp9417", "report", "tables")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    json_path = os.path.join(
        results_dir,
        f"xrfm_interpretability_{dataset_name}.json"
    )
    figure_path = os.path.join(
        figures_dir,
        f"{dataset_name}_feature_importance_comparison.pdf"
    )
    table_path = os.path.join(
        tables_dir,
        f"{dataset_name}_feature_importance_table.tex"
    )

    output = {
        "model": "xrfm",
        "dataset": dataset_name,
        "task_type": task_type,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_time": float(train_time),
        "test_metrics": test_metrics,
        "top_k": top_k,
        "importance_by_method": importance_by_method,
        "top_features_by_method": top_features_by_method
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    plot_importance_comparison(
        importance_by_method=importance_by_method,
        dataset_name=dataset_name,
        save_path=figure_path,
        top_k=top_k
    )

    save_latex_table(
        top_features_by_method=top_features_by_method,
        dataset_name=dataset_name,
        save_path=table_path
    )

    print("\n=== Completed ===")
    print(f"Train time: {train_time:.4f} s")
    print(f"Test metrics: {test_metrics}")
    print(f"Saved JSON to: {json_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved LaTeX table to: {table_path}")

    print("\nTop features by method:")
    for method, features in top_features_by_method.items():
        print(f"\n{method}:")
        for feature, score in features[:5]:
            print(f"  {feature}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="diamonds")
    parser.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_perm_samples", type=int, default=2000)
    parser.add_argument("--n_repeats", type=int, default=5)

    args = parser.parse_args()

    run_interpretability_experiment(
        dataset_name=args.dataset,
        task_type=args.task_type,
        top_k=args.top_k,
        max_perm_samples=args.max_perm_samples,
        n_repeats=args.n_repeats
    )


if __name__ == "__main__":
    main()