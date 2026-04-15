"""Competition evaluation metrics for the Broad Obesity challenge.

Metric specification
--------------------
* pearson_delta  — Pearson correlation of per-gene mean delta vectors
                   (perturbation mean − control mean).
* mmd_score      — Maximum Mean Discrepancy with RBF kernel (multi-bandwidth).
                   Bandwidths: 581.5, 1163, 2326, 4652, 9304.
* l1_distance    — Mean absolute difference between the predicted mean
                   expression and the ground-truth perturbed mean.
* combined_score — 0.75 * S_R + 0.25 * S_L
                   where S_R = similarity vs real perturbed cells
                   and   S_L = similarity vs control cells (difficulty proxy).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from typing import Optional, Sequence

# RBF bandwidths from competition specification
_MMD_BANDWIDTHS: tuple[float, ...] = (581.5, 1163.0, 2326.0, 4652.0, 9304.0)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def pearson_delta(
    pred: np.ndarray,
    true_perturbed: np.ndarray,
    control_mean: np.ndarray,
) -> float:
    """Pearson correlation between predicted and true mean-shift vectors.

    Parameters
    ----------
    pred:
        Predicted cell matrix, shape ``(n_pred_cells, n_genes)``.
    true_perturbed:
        Ground-truth perturbed cells, shape ``(n_true_cells, n_genes)``.
    control_mean:
        Mean expression of control cells, shape ``(n_genes,)``.

    Returns
    -------
    Pearson r ∈ [-1, 1].  Returns 0.0 when either delta is constant.
    """
    pred_delta = pred.mean(axis=0) - control_mean
    true_delta = true_perturbed.mean(axis=0) - control_mean

    if np.std(pred_delta) == 0 or np.std(true_delta) == 0:
        return 0.0

    r, _ = pearsonr(pred_delta, true_delta)
    return float(r)


def _rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidths: Sequence[float] = _MMD_BANDWIDTHS,
) -> np.ndarray:
    """Sum of RBF kernels over multiple bandwidths.

    Parameters
    ----------
    X, Y:
        2-D arrays of shape ``(n, d)`` and ``(m, d)``.
    bandwidths:
        Sequence of bandwidth values σ (NOT σ²).

    Returns
    -------
    Kernel matrix of shape ``(n, m)``.
    """
    # Squared Euclidean distances  ||x - y||²
    XX = (X ** 2).sum(axis=1, keepdims=True)  # (n, 1)
    YY = (Y ** 2).sum(axis=1, keepdims=True)  # (m, 1)
    sq_dist = XX + YY.T - 2 * X @ Y.T         # (n, m)

    K = np.zeros_like(sq_dist)
    for sigma in bandwidths:
        K += np.exp(-sq_dist / (2 * sigma ** 2))
    return K


def mmd_score(
    pred: np.ndarray,
    true_perturbed: np.ndarray,
    bandwidths: Sequence[float] = _MMD_BANDWIDTHS,
    max_cells: int = 1000,
    seed: int = 42,
) -> float:
    """Maximum Mean Discrepancy (lower is better) between pred and true.

    Uses the unbiased U-statistic estimator with multi-bandwidth RBF kernels.

    Parameters
    ----------
    pred:
        Predicted expression matrix, shape ``(n, n_genes)``.
    true_perturbed:
        Ground-truth perturbed expression matrix, shape ``(m, n_genes)``.
    bandwidths:
        RBF bandwidths σ.
    max_cells:
        Sub-sample each matrix to at most this many cells for tractability.
    seed:
        RNG seed for sub-sampling.

    Returns
    -------
    MMD² estimate ≥ 0.  (Returns 0 when either set has < 2 cells.)
    """
    pred = np.asarray(pred, dtype=np.float64)
    true_perturbed = np.asarray(true_perturbed, dtype=np.float64)

    rng = np.random.default_rng(seed)

    def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.shape[0] <= n:
            return arr
        idx = rng.choice(arr.shape[0], size=n, replace=False)
        return arr[idx]

    pred = _subsample(pred, max_cells)
    true_perturbed = _subsample(true_perturbed, max_cells)

    n = pred.shape[0]
    m = true_perturbed.shape[0]

    if n < 2 or m < 2:
        return 0.0

    K_pp = _rbf_kernel(pred, pred, bandwidths)
    K_tt = _rbf_kernel(true_perturbed, true_perturbed, bandwidths)
    K_pt = _rbf_kernel(pred, true_perturbed, bandwidths)

    # Unbiased estimator: zero out diagonals
    np.fill_diagonal(K_pp, 0)
    np.fill_diagonal(K_tt, 0)

    mmd2 = (
        K_pp.sum() / (n * (n - 1))
        + K_tt.sum() / (m * (m - 1))
        - 2 * K_pt.mean()
    )
    return float(max(mmd2, 0.0))


def l1_distance(
    pred: np.ndarray,
    true_perturbed: np.ndarray,
) -> float:
    """Mean absolute error between predicted and true mean expression.

    Parameters
    ----------
    pred:
        Predicted expression matrix, shape ``(n, n_genes)``.
    true_perturbed:
        Ground-truth perturbed matrix, shape ``(m, n_genes)``.

    Returns
    -------
    Mean L1 distance between the two mean vectors (scalar ≥ 0).
    """
    pred_mean = np.asarray(pred, dtype=np.float64).mean(axis=0)
    true_mean = np.asarray(true_perturbed, dtype=np.float64).mean(axis=0)
    return float(np.abs(pred_mean - true_mean).mean())


# ---------------------------------------------------------------------------
# Combined competition score
# ---------------------------------------------------------------------------

def combined_score(
    pred: np.ndarray,
    true_perturbed: np.ndarray,
    control_cells: np.ndarray,
    bandwidths: Sequence[float] = _MMD_BANDWIDTHS,
    w_real: float = 0.75,
    w_ctrl: float = 0.25,
) -> dict[str, float]:
    """Compute the composite competition score.

    Score = w_real * S_R + w_ctrl * S_L

    where:
      S_R = pearson_delta(pred, true_perturbed, control_mean)
            - mmd_score(pred, true_perturbed)
            - l1_distance(pred, true_perturbed)
      S_L = pearson_delta(pred, control_cells, control_mean)
            - mmd_score(pred, control_cells)
            - l1_distance(pred, control_cells)

    Parameters
    ----------
    pred:
        Predicted expression matrix, shape ``(n, n_genes)``.
    true_perturbed:
        Ground-truth perturbed cells, shape ``(m, n_genes)``.
    control_cells:
        Unperturbed / control cells, shape ``(k, n_genes)``.
    bandwidths:
        RBF bandwidths for MMD.
    w_real:
        Weight for real-data term (default 0.75).
    w_ctrl:
        Weight for control-data term (default 0.25).

    Returns
    -------
    Dict with keys: ``score``, ``pearson_r``, ``mmd_r``, ``l1_r``,
    ``pearson_l``, ``mmd_l``, ``l1_l``, ``S_R``, ``S_L``.
    """
    control_mean = np.asarray(control_cells, dtype=np.float64).mean(axis=0)

    # Real-data scores
    pr_r = pearson_delta(pred, true_perturbed, control_mean)
    mmd_r = mmd_score(pred, true_perturbed, bandwidths=bandwidths)
    l1_r = l1_distance(pred, true_perturbed)
    S_R = pr_r - mmd_r - l1_r

    # Control-data scores (difficulty proxy)
    pr_l = pearson_delta(pred, control_cells, control_mean)
    mmd_l = mmd_score(pred, control_cells, bandwidths=bandwidths)
    l1_l = l1_distance(pred, control_cells)
    S_L = pr_l - mmd_l - l1_l

    score = w_real * S_R + w_ctrl * S_L

    return {
        "score": score,
        "S_R": S_R,
        "S_L": S_L,
        "pearson_r": pr_r,
        "mmd_r": mmd_r,
        "l1_r": l1_r,
        "pearson_l": pr_l,
        "mmd_l": mmd_l,
        "l1_l": l1_l,
    }
