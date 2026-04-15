"""Utilities for writing competition submission files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_prediction_h5ad(
    predictions: np.ndarray,
    template_adata: ad.AnnData,
    output_path: str | Path,
    obs_subset: Optional[pd.Index] = None,
) -> None:
    """Write per-cell expression predictions as an h5ad file.

    Parameters
    ----------
    predictions:
        Array of shape ``(n_cells, n_genes)`` with predicted expression.
    template_adata:
        Source AnnData used to copy ``obs`` metadata and ``var`` index.
        Must have the same number of cells as ``predictions`` (or match
        ``obs_subset`` if provided).
    output_path:
        Destination ``.h5ad`` path.
    obs_subset:
        Optional pandas Index selecting which rows of *template_adata* to
        use (e.g. after a train/val split).  When None, the full *template_adata*
        ``obs`` is used.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if obs_subset is not None:
        obs_df = template_adata.obs.loc[obs_subset].copy()
    else:
        obs_df = template_adata.obs.copy()

    if len(obs_df) != predictions.shape[0]:
        raise ValueError(
            f"predictions has {predictions.shape[0]} rows but obs has {len(obs_df)} rows."
        )

    out = ad.AnnData(
        X=predictions.astype(np.float32),
        obs=obs_df,
        var=template_adata.var.copy(),
    )
    out.write_h5ad(output_path)
    logger.info("Wrote predictions h5ad to %s  (%d cells)", output_path, out.n_obs)


def write_program_proportions(
    proportions: np.ndarray,
    cell_ids: pd.Index,
    output_path: str | Path,
    program_prefix: str = "program",
) -> None:
    """Write gene-program proportions as a CSV file.

    Parameters
    ----------
    proportions:
        Array of shape ``(n_cells, n_programs)`` with non-negative proportions
        that sum to 1 per row.
    cell_ids:
        Cell identifiers (e.g. ``adata.obs_names``) for the row index.
    output_path:
        Destination ``.csv`` path.
    program_prefix:
        Column name prefix; columns will be ``program_0``, ``program_1``, …
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(cell_ids) != proportions.shape[0]:
        raise ValueError(
            f"proportions has {proportions.shape[0]} rows but cell_ids has {len(cell_ids)} entries."
        )

    n_programs = proportions.shape[1]
    cols = [f"{program_prefix}_{i}" for i in range(n_programs)]
    df = pd.DataFrame(proportions, index=cell_ids, columns=cols)
    df.index.name = "cell_id"
    df.to_csv(output_path)
    logger.info(
        "Wrote program proportions to %s  (%d cells, %d programs)",
        output_path,
        len(cell_ids),
        n_programs,
    )


def write_method_description(
    output_path: str | Path,
    method_name: str,
    description: str,
    hyperparameters: Optional[dict] = None,
) -> None:
    """Write a plain-text method description for the submission.

    Parameters
    ----------
    output_path:
        Destination text / JSON path.
    method_name:
        Short name for the method (e.g. ``"MeanShiftVAE"``).
    description:
        Free-text description of the approach.
    hyperparameters:
        Optional dict of key hyperparameters to record.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "method_name": method_name,
        "description": description,
    }
    if hyperparameters:
        payload["hyperparameters"] = hyperparameters

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote method description to %s", output_path)
