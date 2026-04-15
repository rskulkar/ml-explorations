"""Linear perturbation baseline predictor."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

from broad_obesity.data.loader import _dense

logger = logging.getLogger(__name__)


class LinearPerturbationPredictor:
    """Ridge-regression baseline: predict expression from perturbation identity.

    Each perturbation is one-hot encoded; Ridge regression maps that
    indicator to the gene-expression space.  The result is equivalent to
    per-perturbation means with L2 regularisation (useful when some
    perturbations have very few cells).

    Parameters
    ----------
    perturbation_col:
        ``adata.obs`` column carrying the perturbation label.
    alpha:
        Ridge regularisation strength.
    """

    def __init__(
        self,
        perturbation_col: str = "perturbation",
        alpha: float = 1.0,
    ) -> None:
        self.perturbation_col = perturbation_col
        self.alpha = alpha

        self._label_enc: Optional[LabelEncoder] = None
        self._ridge: Optional[Ridge] = None
        self.gene_names_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata: ad.AnnData) -> "LinearPerturbationPredictor":
        """Fit the Ridge regressor on *adata*.

        Parameters
        ----------
        adata:
            Training AnnData (cells × genes).

        Returns
        -------
        self
        """
        logger.info(
            "Fitting LinearPerturbationPredictor on %d cells × %d genes",
            adata.n_obs,
            adata.n_vars,
        )
        self.gene_names_ = np.asarray(adata.var_names)
        labels = adata.obs[self.perturbation_col].values

        self._label_enc = LabelEncoder()
        encoded = self._label_enc.fit_transform(labels)  # (n_cells,)

        # One-hot feature matrix  (n_cells, n_perturbations)
        n_classes = len(self._label_enc.classes_)
        X_ohe = np.zeros((len(encoded), n_classes), dtype=np.float32)
        X_ohe[np.arange(len(encoded)), encoded] = 1.0

        Y = _dense(adata.X)  # (n_cells, n_genes)

        self._ridge = Ridge(alpha=self.alpha)
        self._ridge.fit(X_ohe, Y)
        logger.info(
            "Ridge fit complete; %d classes", n_classes
        )
        return self

    def predict(self, adata: ad.AnnData) -> np.ndarray:
        """Predict expression for cells in *adata*.

        Parameters
        ----------
        adata:
            AnnData whose ``obs[perturbation_col]`` specifies each cell's
            perturbation label.

        Returns
        -------
        Array of shape ``(n_cells, n_genes)`` with float32 predictions.
        """
        if self._ridge is None or self._label_enc is None:
            raise RuntimeError("Call fit() before predict().")

        labels = adata.obs[self.perturbation_col].values
        n_classes = len(self._label_enc.classes_)

        # Map unseen labels to class 0 as fallback (log a warning)
        encoded = np.zeros(len(labels), dtype=int)
        unknown: set[str] = set()
        for i, lbl in enumerate(labels):
            if lbl in self._label_enc.classes_:
                encoded[i] = self._label_enc.transform([lbl])[0]
            else:
                unknown.add(str(lbl))
                encoded[i] = 0

        if unknown:
            logger.warning(
                "Unseen perturbations mapped to class 0: %s",
                ", ".join(sorted(unknown)),
            )

        X_ohe = np.zeros((len(encoded), n_classes), dtype=np.float32)
        X_ohe[np.arange(len(encoded)), encoded] = 1.0

        preds = self._ridge.predict(X_ohe).astype(np.float32)
        return preds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise to *path* (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved LinearPerturbationPredictor to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearPerturbationPredictor":
        """Deserialise from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}")
        logger.info("Loaded LinearPerturbationPredictor from %s", path)
        return obj
