"""Perturbed-mean baseline predictor."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd

from broad_obesity.data.loader import (
    _dense,
    compute_perturbation_means,
)

logger = logging.getLogger(__name__)


class PerturbedMeanPredictor:
    """Predicts each cell's expression as the training mean for its perturbation.

    This is the simplest meaningful baseline for perturbation prediction:
    given a held-out cell that received perturbation *p*, predict the
    average gene-expression profile observed for *p* during training.

    Parameters
    ----------
    perturbation_col:
        ``adata.obs`` column that carries the perturbation label.
    control_label:
        Label used for unperturbed / control cells.  The control mean is
        used as fallback for unseen perturbations at inference time.
    """

    def __init__(
        self,
        perturbation_col: str = "perturbation",
        control_label: str = "control",
    ) -> None:
        self.perturbation_col = perturbation_col
        self.control_label = control_label

        # Populated by fit()
        self._means: Optional[pd.DataFrame] = None
        self._control_mean: Optional[np.ndarray] = None
        self.gene_names_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata: ad.AnnData) -> "PerturbedMeanPredictor":
        """Compute per-perturbation means from training data.

        Parameters
        ----------
        adata:
            Training AnnData (cells × genes).

        Returns
        -------
        self
        """
        logger.info("Fitting PerturbedMeanPredictor on %d cells", adata.n_obs)
        self._means = compute_perturbation_means(
            adata, perturbation_col=self.perturbation_col
        )
        self.gene_names_ = np.asarray(adata.var_names)

        if self.control_label in self._means.index:
            self._control_mean = self._means.loc[self.control_label].values.astype(
                np.float32
            )
        else:
            logger.warning(
                "Control label '%s' not found; using grand mean as fallback.",
                self.control_label,
            )
            self._control_mean = (
                _dense(adata.X).mean(axis=0).astype(np.float32)
            )

        logger.info(
            "Stored means for %d perturbations", len(self._means)
        )
        return self

    def predict(self, adata: ad.AnnData) -> np.ndarray:
        """Return predicted expression for each cell in *adata*.

        Parameters
        ----------
        adata:
            AnnData whose ``obs[perturbation_col]`` specifies the perturbation
            that each cell received.  ``adata.var_names`` must match training.

        Returns
        -------
        Array of shape ``(n_cells, n_genes)`` with float32 predictions.
        """
        if self._means is None:
            raise RuntimeError("Call fit() before predict().")

        labels = adata.obs[self.perturbation_col].values
        n_genes = len(self.gene_names_)
        out = np.empty((len(labels), n_genes), dtype=np.float32)

        missing: set[str] = set()
        for i, label in enumerate(labels):
            if label in self._means.index:
                out[i] = self._means.loc[label].values.astype(np.float32)
            else:
                missing.add(str(label))
                out[i] = self._control_mean

        if missing:
            logger.warning(
                "Unseen perturbations (using control mean): %s",
                ", ".join(sorted(missing)),
            )
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the predictor to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved PerturbedMeanPredictor to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PerturbedMeanPredictor":
        """Deserialise from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}")
        logger.info("Loaded PerturbedMeanPredictor from %s", path)
        return obj
