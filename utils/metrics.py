import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if y_prob is not None:
        y_prob = np.array(y_prob).flatten()
        try:
            result["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            result["auc"] = None
    else:
        result["auc"] = None
    return result


def calculate_feature_importance_baselines(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    feature_names: list,
    task_type: str,
    model = None
) -> dict:
    result = {}
    n_features = X_train.shape[1]
    pca = PCA(n_components=1, random_state=42)
    pca.fit(X_train)
    pca_loadings = np.abs(pca.components_[0])
    result["pca"] = dict(zip(feature_names, pca_loadings.tolist()))
    if task_type == "regression":
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    else:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    result["mutual_info"] = dict(zip(feature_names, mi_scores.tolist()))
    if model is not None:
        perm_imp = permutation_importance(
            model, X_train, y_train, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        result["permutation_importance"] = dict(zip(feature_names, perm_imp.importances_mean.tolist()))
    return result
