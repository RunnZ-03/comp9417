import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def process_dataset(filepath, target_col, task_type, drop_cols=None):
    df = pd.read_csv(filepath)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols_raw = X.select_dtypes(exclude=['number']).columns.tolist()
    for col in cat_cols_raw:
        X[col] = X[col].astype(object)
        X[col] = X[col].fillna(np.nan)

    if task_type == 'classification':
        if y.dtype == 'object' or str(y.dtype) == 'boolean' or y.dtype == 'bool' or str(y.dtype).startswith('string'):
            y = (y == y.unique()[0]).astype(int)

    if task_type == 'classification':
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42,
                                                          stratify=y_temp)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feat_names = preprocessor.get_feature_names_out()

    return {
        'X_train': X_train_processed, 'y_train': y_train.values,
        'X_val': X_val_processed, 'y_val': y_val.values,
        'X_test': X_test_processed, 'y_test': y_test.values,
        'features': feat_names
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

datasets_config = {
    "diamonds": {
        "filepath": os.path.join(DATA_DIR, "Diamonds Prices2022.csv"),
        "target": "price",
        "task": "regression",
        "drop": ["Unnamed: 0"]
    },
    "superconductivity": {
        "filepath": os.path.join(DATA_DIR, "train.csv"),
        "target": "critical_temp",
        "task": "regression",
        "drop": None
    },
    "stroke": {
        "filepath": os.path.join(DATA_DIR, "healthcare-dataset-stroke-data.csv"),
        "target": "stroke",
        "task": "classification",
        "drop": ["id"]
    },
    "shoppers": {
        "filepath": os.path.join(DATA_DIR, "10.0online_shoppers_intention.csv"),
        "target": "Revenue",
        "task": "classification",
        "drop": None
    },
    "hr_attrition": {
        "filepath": os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        "target": "Attrition",
        "task": "classification",
        "drop": ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    }
}

processed_data = {}

for dataset_name, config in datasets_config.items():
    try:
        processed_data[dataset_name] = process_dataset(
            filepath=config["filepath"],
            target_col=config["target"],
            task_type=config["task"],
            drop_cols=config["drop"]
        )
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")


def get_dataset(name):
    if name not in processed_data:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(processed_data.keys())}")

    data = processed_data[name]
    return (
        data['X_train'], data['X_val'], data['X_test'],
        data['y_train'], data['y_val'], data['y_test'],
        data['features']
    )

def get_dataset_subsampled(name, sample_size_fraction=None, sample_size_absolute=None):
    data = processed_data[name]
    X_train_full = data['X_train']
    y_train_full = data['y_train']

    if sample_size_fraction is not None:
        n_samples = max(100, int(len(X_train_full) * sample_size_fraction))
    elif sample_size_absolute is not None:
        n_samples = min(len(X_train_full), max(100, sample_size_absolute))
    else:
        return (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test'],
            data['features']
        )

    np.random.seed(42)
    indices = np.random.choice(len(X_train_full), n_samples, replace=False)
    X_train_sub = X_train_full[indices]
    y_train_sub = y_train_full[indices]

    return (
        X_train_sub, data['X_val'], data['X_test'],
        y_train_sub, data['y_val'], data['y_test'],
        data['features']
    )

if __name__ == "__main__":
    print("--- finished ---")
    for name in datasets_config.keys():
        if name in processed_data:
            X_tr = processed_data[name]['X_train']
            y_tr = processed_data[name]['y_train']
            print(f"✅ {name:18} | X_train: {X_tr.shape}, y_train: {y_tr.shape}")
