"""
bonus/residual_agop.py
======================
Residual-weighted AGOP extension (Bonus).

Standard AGOP:
    G_f = (1/n) sum_i  nabla f(x_i) nabla f(x_i)^T

Residual-weighted AGOP  [phi(r) = r^2]:
    AGOP_res(f) = sum_i r_i^2 * nabla f(x_i) nabla f(x_i)^T
                  -----------------------------------------
                              sum_i r_i^2

Usage:
    python bonus/residual_agop.py
"""

import os, sys, warnings
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
COMP9417_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if COMP9417_ROOT not in sys.path:
    sys.path.insert(0, COMP9417_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_model(alpha=2.0):
    return make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                         Ridge(alpha=alpha))


def grad_matrix(model, X, eps=1e-3):
    """Forward finite-difference gradient  nabla f(x_i) for every row."""
    n, d = X.shape
    base = model.predict(X)
    G = np.zeros((n, d))
    for j in range(d):
        Xp = X.copy(); Xp[:, j] += eps
        G[:, j] = (model.predict(Xp) - base) / eps
    return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)


def agop(G, residuals=None):
    """
    G         : (n, d) gradient matrix
    residuals : None → standard (uniform weights); else residual-weighted phi(r)=r^2
    Returns   : (d, d) AGOP matrix
    """
    w = np.ones(len(G)) / len(G) if residuals is None else residuals**2
    w = w / (w.sum() + 1e-12)
    return np.einsum('i,ij,ik->jk', w, G, G)


def top_eigvec(M):
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, -1], float(vals[-1])


def cosine(u, v):
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def best_split(X, y, v, n_cand=30):
    proj = X @ v
    t, best = float(np.median(proj)), np.inf
    for c in np.percentile(proj, np.linspace(10, 90, n_cand)):
        l, r = proj <= c, proj > c
        if l.sum() < 5 or r.sum() < 5:
            continue
        s = (np.var(y[l]) * l.sum() + np.var(y[r]) * r.sum()) / len(y)
        if s < best:
            best, t = s, c
    return t


def eval_split(Xtr, ytr, Xte, yte, v, t, alpha=2.0):
    """Split Xtr/ytr along v/t, fit poly-2 Ridge leaves, return test RMSE."""
    ptr, pte = Xtr @ v, Xte @ v
    preds = np.full(len(yte), ytr.mean())
    for mtr, mte in [(ptr <= t, pte <= t), (ptr > t, pte > t)]:
        if mtr.sum() >= 8 and mte.sum() > 0:
            leaf = make_model(alpha); leaf.fit(Xtr[mtr], ytr[mtr])
            preds[mte] = leaf.predict(Xte[mte])
    return float(np.sqrt(np.mean((preds - yte)**2)))


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_bonus():
    print("=" * 64)
    print("Residual-Weighted AGOP  —  Bonus Extension")
    print("=" * 64)

    rng = np.random.default_rng(42)
    d = 6

    # ── (ii) Basic implementation on single-regime dataset ────────────────────
    print("\n[1] Implementation  n=200, d=6")
    print("    y = 4·x0 + 3·sin(2·x1) + 0.5·eps")
    X1 = rng.standard_normal((200, d))
    y1 = 4*X1[:,0] + 3*np.sin(2*X1[:,1]) + 0.5*rng.standard_normal(200)
    m1 = make_model(); m1.fit(X1, y1)
    res1 = y1 - m1.predict(X1)
    G1 = grad_matrix(m1, X1)
    Gs1, Gr1 = agop(G1), agop(G1, res1)
    vs1, _ = top_eigvec(Gs1); vr1, _ = top_eigvec(Gr1)
    print(f"  Std  top-2 features: {sorted(np.argsort(np.abs(vs1))[::-1][:2])}")
    print(f"  Res  top-2 features: {sorted(np.argsort(np.abs(vr1))[::-1][:2])}")
    print(f"  Cosine similarity  : {cosine(vs1, vr1):.4f}")
    print(f"  Mean |residual|    : {np.mean(np.abs(res1)):.4f}")

    # ── (iii) Disagreement: two-regime dataset ────────────────────────────────
    # Regime A (x5<0): y = 5·x0, tiny noise  → model fits well → small residuals
    # Regime B (x5≥0): y = 5·x2, large noise → model fits poorly → large residuals
    # Standard AGOP: x0 dominates (well-fitted signal)
    # Residual-weighted AGOP: x2 gains weight (hard-to-fit regime is upweighted)
    print("\n[2] Disagreement  n=600, d=6")
    print("    Regime A (x5<0): y=5·x0 + 0.05·eps  [well-fitted, small residuals]")
    print("    Regime B (x5≥0): y=5·x2 + 5.0·eps   [poorly-fitted, large residuals]")
    X2 = rng.standard_normal((600, d))
    rA = X2[:,5] < 0
    eps2 = rng.standard_normal(600)
    y2 = np.where(rA, 5*X2[:,0] + 0.05*eps2, 5*X2[:,2] + 5.0*eps2)

    m2 = make_model(alpha=0.5); m2.fit(X2, y2)
    res2 = y2 - m2.predict(X2)
    G2   = grad_matrix(m2, X2)
    Gs2, Gr2 = agop(G2), agop(G2, res2)
    vs2, _ = top_eigvec(Gs2); vr2, _ = top_eigvec(Gr2)

    top1s = int(np.argmax(np.abs(vs2)))
    top1r = int(np.argmax(np.abs(vr2)))
    cs2   = cosine(vs2, vr2)
    rmA   = float(np.sqrt(np.mean(res2[rA]**2)))
    rmB   = float(np.sqrt(np.mean(res2[~rA]**2)))

    print(f"  Residual RMSE regime A (well-fit) : {rmA:.3f}")
    print(f"  Residual RMSE regime B (poor-fit) : {rmB:.3f}")
    print(f"  Std  top feature: x{top1s}   loading={abs(vs2[top1s]):.3f}")
    print(f"  Res  top feature: x{top1r}   loading={abs(vr2[top1r]):.3f}")
    print(f"  Cosine similarity            : {cs2:.4f}")
    print(f"  Top features differ          : {top1s != top1r}")
    print(f"  Std  eigvec abs[:3]: {np.round(np.abs(vs2[:3]),3)}")
    print(f"  Res  eigvec abs[:3]: {np.round(np.abs(vr2[:3]),3)}")

    # ── (iv) Performance: same two-regime DGP ────────────────────────────────
    print("\n[3] Performance  n_train=500, n_test=200")
    Xp = rng.standard_normal((700, d))
    rAp = Xp[:,5] < 0
    ep  = rng.standard_normal(700)
    yp  = np.where(rAp, 5*Xp[:,0]+0.05*ep, 5*Xp[:,2]+5.0*ep)
    Xtr, Xte = Xp[:500], Xp[500:]
    ytr, yte  = yp[:500], yp[500:]

    m3 = make_model(alpha=0.5); m3.fit(Xtr, ytr)
    res3 = ytr - m3.predict(Xtr)
    G3   = grad_matrix(m3, Xtr)
    Gs3, Gr3 = agop(G3), agop(G3, res3)
    vs3, _ = top_eigvec(Gs3); vr3, _ = top_eigvec(Gr3)

    ts = best_split(Xtr, ytr, vs3)
    tr = best_split(Xtr, ytr, vr3)
    rmse_global = float(np.sqrt(np.mean((m3.predict(Xte) - yte)**2)))
    rmse_std    = eval_split(Xtr, ytr, Xte, yte, vs3, ts)
    rmse_res    = eval_split(Xtr, ytr, Xte, yte, vr3, tr)

    print(f"  Global model RMSE              : {rmse_global:.4f}")
    print(f"  Standard  AGOP split RMSE      : {rmse_std:.4f}")
    print(f"  Res-weighted AGOP split RMSE   : {rmse_res:.4f}")
    print(f"  RMSE reduction (std -> res)    : {rmse_std - rmse_res:+.4f}")

    print("\nBonus complete.\n")
    return dict(
        impl_cosim   = round(cosine(vs1, vr1), 4),
        dis_top_std  = f"x{top1s}", dis_top_res = f"x{top1r}",
        dis_cosim    = round(cs2, 4),   dis_differ = top1s != top1r,
        rmse_global  = round(rmse_global, 4),
        rmse_std_spl = round(rmse_std,    4),
        rmse_res_spl = round(rmse_res,    4),
    )


if __name__ == "__main__":
    r = run_bonus()
    print("Summary:", r)
