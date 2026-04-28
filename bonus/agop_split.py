"""
bonus/agop_split.py
===================
Scratch implementation of the AGOP-based splitting criterion.

For a linear model  f(x) = w^T x + b  the gradient is constant:
    grad f(x) = w   for all x
so the AGOP matrix simplifies to:
    G_f = E[grad f(x) grad f(x)^T] = w w^T
and its diagonal gives coordinate-wise importance: [G_f]_ii = w_i^2.
The leading eigenvector of G_f is  w / ||w||  — the split direction.

We verify this on a synthetic dataset where only x_0 and x_1 carry signal
(y = 3 x_0 - 2 x_1 + noise), then compare with xRFM's AGOP ranking.

Usage:
    python bonus/agop_split.py
"""

import os
import sys
import numpy as np
from sklearn.linear_model import Ridge

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
COMP9417_ROOT  = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_PARENT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)


# ─────────────────────────────────────────────────────────────────────────────
# Scratch AGOP splitting criterion
# ─────────────────────────────────────────────────────────────────────────────

class ScratchAGOPSplit:
    """
    AGOP splitting criterion implemented from scratch using NumPy + scikit-learn.

    Model: Ridge regression  f(x) = w^T x + b
    Gradient: grad f(x) = w  (constant — does not depend on x)
    AGOP matrix: G_f = w w^T  in R^{d x d}
    AGOP diagonal: [G_f]_ii = w_i^2  (coordinate-wise feature importance)
    Split direction: v = w / ||w||  (leading eigenvector of rank-1 G_f)
    Split threshold: argmin_{t} variance of y in each half of {x: v^T x <= t}
    """

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchAGOPSplit":
        self.model.fit(X, y)
        self.w_ = self.model.coef_          # shape (d,)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # ── AGOP ─────────────────────────────────────────────────────────────────

    def agop_matrix(self) -> np.ndarray:
        """G_f = w w^T   (exact, no sampling needed for linear model)."""
        return np.outer(self.w_, self.w_)

    def agop_diagonal(self) -> np.ndarray:
        """Diagonal of AGOP = [w_i^2].  Feature importance scores."""
        return self.w_ ** 2

    # ── split direction ───────────────────────────────────────────────────────

    def split_direction(self):
        """v = w / ||w||  (unique leading eigenvector of rank-1 AGOP)."""
        norm = np.linalg.norm(self.w_)
        return self.w_ / (norm + 1e-12), norm ** 2   # (direction, eigenvalue)

    # ── optimal threshold ─────────────────────────────────────────────────────

    def find_split(self, X: np.ndarray, y: np.ndarray):
        """Scan candidate thresholds along v; pick minimum-variance split."""
        v, eigval = self.split_direction()
        proj      = X @ v
        candidates = np.percentile(proj, np.linspace(10, 90, 30))
        best_t, best_score = None, np.inf
        for t in candidates:
            lm, rm = proj <= t, proj > t
            if lm.sum() < 2 or rm.sum() < 2:
                continue
            score = (np.var(y[lm]) * lm.sum() + np.var(y[rm]) * rm.sum()) / len(y)
            if score < best_score:
                best_score, best_t = score, t
        return v, best_t, eigval


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))


def run_verification():
    """
    Synthetic dataset: y = 3 x_0 - 2 x_1 + noise  (n=300, d=10).
    Ground-truth signal lives in features {0, 1}.
    Both the scratch criterion and the xRFM library should rank x_0 and x_1
    at the top — confirming that the scratch criterion recovers the correct
    split direction.
    """
    from xrfm import xRFM

    print("=" * 64)
    print("AGOP Split Criterion  —  Scratch vs xRFM Verification")
    print("Dataset : Synthetic  n=300, d=10,  y = 3·x₀ − 2·x₁ + ε")
    print("=" * 64)

    # ── synthetic data ────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n, d = 300, 10
    X = rng.standard_normal((n, d))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.3 * rng.standard_normal(n)
    feat_names = [f"x{i}" for i in range(d)]

    n_val = 60
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    # ── scratch AGOP ──────────────────────────────────────────────────────────
    print("\n[1] Scratch model (Ridge regression) — analytical AGOP")
    scratch = ScratchAGOPSplit(alpha=1e-3)
    scratch.fit(X_tr, y_tr)

    diag    = scratch.agop_diagonal()
    v, t, ev = scratch.find_split(X_tr, y_tr)
    top2_scratch = set(np.argsort(diag)[::-1][:2])

    print(f"  Fitted weights  w: {np.round(scratch.w_, 3)}")
    print(f"  AGOP diagonal w²: {np.round(diag, 3)}")
    print(f"  Top-2 features  : {sorted(top2_scratch)}")
    print(f"  Split direction v (≈ w/‖w‖): {np.round(v[:4], 3)} …")
    print(f"  Eigenvalue      : {ev:.4f}")
    print(f"  Split threshold : {t:.4f}")

    # ── xRFM library — numerical AGOP diagonal ────────────────────────────────
    print("\n[2] xRFM library — numerical AGOP diagonal (black-box predictions)")
    lib = xRFM(use_diag=True, device="cpu")
    lib.fit(X_tr, y_tr, X_val, y_val)

    eps       = 1e-3
    base_pred = lib.predict(X_tr)
    diag_lib  = np.zeros(d)
    for j in range(d):
        dX = np.zeros_like(X_tr); dX[:, j] = eps
        grad_j = (lib.predict(X_tr + dX) - base_pred) / eps
        diag_lib[j] = float(np.mean(grad_j ** 2))

    top2_lib = set(np.argsort(diag_lib)[::-1][:2])
    print(f"  Numerical AGOP diagonal: {np.round(diag_lib, 4)}")
    print(f"  Top-2 features         : {sorted(top2_lib)}")

    # ── agreement ─────────────────────────────────────────────────────────────
    overlap = len(top2_scratch & top2_lib)
    cosim   = _cosine_sim(diag    / (diag.sum()     + 1e-12),
                          diag_lib / (diag_lib.sum() + 1e-12))

    correct_scratch = top2_scratch == {0, 1}
    correct_lib     = top2_lib     == {0, 1}

    print(f"\n[3] Agreement")
    print(f"  Ground-truth signal features : {{x0, x1}}")
    print(f"  Scratch identifies {{x0,x1}} : {correct_scratch}")
    print(f"  Library identifies {{x0,x1}} : {correct_lib}")
    print(f"  Top-2 overlap                : {overlap}/2")
    print(f"  Cosine similarity (diagonals): {cosim:.4f}")
    print("\nVerification complete.\n")

    return {
        "scratch_weights"   : scratch.w_.tolist(),
        "scratch_top2"      : sorted(top2_scratch),
        "lib_top2"          : sorted(top2_lib),
        "overlap"           : overlap,
        "cosine_sim"        : round(cosim, 4),
        "correct_scratch"   : correct_scratch,
        "correct_lib"       : correct_lib,
    }


if __name__ == "__main__":
    results = run_verification()
