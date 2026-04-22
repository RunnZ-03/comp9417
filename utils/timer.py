import time
from contextlib import contextmanager
import numpy as np


@contextmanager
def timer():
    start = time.time()
    class TimerResult:
        elapsed = 0.0
    result = TimerResult()
    try:
        yield result
    finally:
        result.elapsed = time.time() - start


def calculate_inference_time(model, X_test: np.ndarray, n_repeats: int = 5) -> dict:
    n_samples = len(X_test)
    if n_samples == 0:
        return {"infer_time_total": 0.0, "infer_time_per_sample": 0.0}
    model.predict(X_test[:1])
    total_times = []
    for _ in range(n_repeats):
        with timer() as t:
            model.predict(X_test)
        total_times.append(t.elapsed)
    infer_time_total = float(np.median(total_times))
    infer_time_per_sample = infer_time_total / n_samples
    return {
        "infer_time_total": infer_time_total,
        "infer_time_per_sample": infer_time_per_sample
    }
