"""Data loading and preprocessing for the Broad Obesity competition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def load_adata(path: str | Path) -> ad.AnnData:
    """Load an AnnData h5ad file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    logger.info("Loading AnnData from %s", path)
    adata = ad.read_h5ad(path)
    logger.info(
        "Loaded %d cells × %d genes; obs keys: %s",
        adata.n_obs,
        adata.n_vars,
        list(adata.obs.columns),
    )
    return adata


def train_val_split(
    adata: ad.AnnData,
    val_frac: float = 0.2,
    perturbation_col: str = "perturbation",
    seed: int = 42,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Stratified train/val split preserving perturbation balance.

    Parameters
    ----------
    adata:
        Full annotated dataset.
    val_frac:
        Fraction of cells to reserve for validation.
    perturbation_col:
        Column in ``adata.obs`` that identifies the perturbation label.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_adata, val_adata)
    """
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []

    for _, grp_idx in adata.obs.groupby(perturbation_col).groups.items():
        grp_idx = list(grp_idx)
        n_val = max(1, int(len(grp_idx) * val_frac))
        chosen = rng.choice(len(grp_idx), size=n_val, replace=False)
        val_indices.extend([grp_idx[i] for i in chosen])

    val_mask = np.zeros(adata.n_obs, dtype=bool)
    val_mask[val_indices] = True

    train_adata = adata[~val_mask].copy()
    val_adata = adata[val_mask].copy()
    logger.info(
        "Split: %d train / %d val cells", train_adata.n_obs, val_adata.n_obs
    )
    return train_adata, val_adata


def _dense(X) -> np.ndarray:
    """Return a dense float32 copy of a matrix (handles sparse)."""
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def compute_perturbation_means(
    adata: ad.AnnData,
    perturbation_col: str = "perturbation",
) -> pd.DataFrame:
    """Compute per-perturbation mean expression across cells.

    Parameters
    ----------
    adata:
        Annotated dataset (cells × genes).
    perturbation_col:
        Column in ``adata.obs`` identifying the perturbation.

    Returns
    -------
    DataFrame of shape (n_perturbations, n_genes) indexed by perturbation label.
    """
    X = _dense(adata.X)
    labels = adata.obs[perturbation_col].values
    unique = np.unique(labels)

    rows = {}
    for pert in unique:
        mask = labels == pert
        rows[pert] = X[mask].mean(axis=0)

    df = pd.DataFrame(rows, index=adata.var_names).T
    df.index.name = perturbation_col
    return df


def compute_perturbed_mean(
    adata: ad.AnnData,
    perturbation: str,
    perturbation_col: str = "perturbation",
) -> np.ndarray:
    """Return the mean expression vector for a single perturbation.

    Parameters
    ----------
    adata:
        Annotated dataset.
    perturbation:
        Perturbation label to query.
    perturbation_col:
        Column in ``adata.obs`` identifying the perturbation.

    Returns
    -------
    1-D array of shape (n_genes,).
    """
    mask = adata.obs[perturbation_col] == perturbation
    if not mask.any():
        raise ValueError(
            f"Perturbation '{perturbation}' not found in column '{perturbation_col}'."
        )
    X = _dense(adata.X)
    return X[mask.values].mean(axis=0)


def compute_program_proportions(
    adata: ad.AnnData,
    n_programs: int = 10,
    seed: int = 42,
    layer: Optional[str] = None,
) -> np.ndarray:
    """Estimate gene-program proportions via NMF.

    Parameters
    ----------
    adata:
        Annotated dataset.
    n_programs:
        Number of latent gene programs.
    seed:
        Random seed for NMF initialisation.
    layer:
        Optional layer key; uses ``adata.X`` if None.

    Returns
    -------
    Array of shape (n_cells, n_programs) with non-negative proportions
    normalised to sum to 1 per cell.
    """
    from sklearn.decomposition import NMF

    X = _dense(adata.layers[layer] if layer else adata.X)
    # Clip negatives that can arise from log-normalised data
    X = np.clip(X, 0, None)

    model = NMF(n_components=n_programs, random_state=seed, max_iter=500)
    W = model.fit_transform(X)  # (n_cells, n_programs)

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    proportions = (W / row_sums).astype(np.float32)
    return proportions
