```markdown
# COMP9417 Group Project: xRFM for Tabular Data

## Project Structure
```
comp9417/
├── data/
├── models/
│   ├── xrfm/
│   ├── xgboost/
│   ├── lightgbm/
│   └── mlp/
├── utils/
│   ├── metrics.py
│   ├── timer.py
│   └── plot_utils.py
├── plots/
├── results/
├── result_summary.py
└── README.md
```

## Unified Tool Usage
### 1. Evaluation Metrics
```python
from utils.metrics import evaluate_regression, evaluate_classification

test_metrics = evaluate_regression(y_test, test_pred)
test_metrics = evaluate_classification(y_test, test_pred, test_prob)
```

### 2. Timing
```python
from utils.timer import timer, calculate_inference_time

with timer() as t:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = t.elapsed

infer_result = calculate_inference_time(model, X_test)
infer_time_total = infer_result["infer_time_total"]
infer_time_per_sample = infer_result["infer_time_per_sample"]
```

### 3. Plotting
```python
from utils.plot_utils import plot_feature_importance_comparison, plot_scalability_curve

plot_feature_importance_comparison(agop_importance, baseline_importance, dataset_name="diamonds")

plot_scalability_curve(
    sample_sizes=[1000, 5000, 10000, 30000],
    performance_results={"xrfm": [0.8, 0.85, 0.88, 0.9], "xgboost": [0.78, 0.83, 0.86, 0.89]},
    time_results={"xrfm": [0.5, 2, 5, 15], "xgboost": [0.3, 1, 3, 10]},
    dataset_name="diamonds",
    task_type="regression",
    performance_metric="RMSE"
)
```

### 4. Result Summary
```bash
python result_summary.py
```

## Datasets
1.  diamonds: Regression
2.  superconductivity: Regression
3.  stroke: Binary Classification
4.  shoppers: Binary Classification
5.  hr_attrition: Binary Classification
```
