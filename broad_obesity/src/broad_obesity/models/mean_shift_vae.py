"""Mean-Shift Variational Autoencoder for perturbation prediction.

Architecture
------------
* GeneEmbedding     — learnable per-gene embeddings used in the decoder.
* Encoder           — maps (expression, perturbation_emb) → (μ, log σ²).
* Decoder           — maps (z, perturbation_emb) → reconstructed expression.
* ProgramProportionHead — maps z → Dirichlet-style program proportions.
* MeanShiftVAE      — assembles the above; ELBO = recon + KL + consistency
                      + proportion regularisation.

Training objective
------------------
  L = reconstruction_loss          # MSE between decoded output and input X
    + β * KL_loss                  # KL(q(z|x,p) || N(0,I))
    + γ * consistency_loss         # ||predicted_mean_shift - true_mean_shift||²
    + δ * proportion_loss          # entropy regularisation on program proportions
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from broad_obesity.data.loader import _dense, compute_perturbation_means

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VAEConfig:
    n_genes: int = 2000
    n_perturbations: int = 100
    latent_dim: int = 64
    hidden_dim: int = 256
    n_programs: int = 10
    pert_emb_dim: int = 32
    beta: float = 1e-3       # KL weight
    gamma: float = 0.1       # consistency weight
    delta: float = 0.01      # proportion entropy weight
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 50
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class GeneEmbedding(nn.Module):
    """Learnable gene-specific weight vector used in the decoder."""

    def __init__(self, n_genes: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_genes, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self) -> torch.Tensor:  # (n_genes, emb_dim)
        idx = torch.arange(self.embedding.num_embeddings, device=self.embedding.weight.device)
        return self.embedding(idx)


class Encoder(nn.Module):
    """q(z | x, perturbation) → (μ, log σ²)."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        inp = cfg.n_genes + cfg.pert_emb_dim
        self.net = nn.Sequential(
            nn.Linear(inp, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim // 2, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim // 2, cfg.latent_dim)

    def forward(
        self, x: torch.Tensor, p_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, p_emb], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """p(x | z, perturbation) → reconstructed expression."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        inp = cfg.latent_dim + cfg.pert_emb_dim
        self.net = nn.Sequential(
            nn.Linear(inp, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.n_genes),
        )

    def forward(self, z: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, p_emb], dim=-1))


class ProgramProportionHead(nn.Module):
    """Maps latent z to normalised gene-program proportions (simplex)."""

    def __init__(self, latent_dim: int, n_programs: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_programs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(z), dim=-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MeanShiftVAE(nn.Module):
    """Mean-Shift VAE with ELBO + consistency + proportion losses."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pert_embedding = nn.Embedding(cfg.n_perturbations, cfg.pert_emb_dim)
        self.gene_embedding = GeneEmbedding(cfg.n_genes, cfg.pert_emb_dim)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.prop_head = ProgramProportionHead(cfg.latent_dim, cfg.n_programs)

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, pert_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        p_emb = self.pert_embedding(pert_idx)          # (B, pert_emb_dim)
        mu, logvar = self.encoder(x, p_emb)
        # Use mean (deterministic) at inference time; sample during training
        if self.training:
            z = self.reparameterise(mu, logvar)
        else:
            z = mu
        x_recon = self.decoder(z, p_emb)
        proportions = self.prop_head(z)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "proportions": proportions,
        }

    # ------------------------------------------------------------------
    # ELBO loss
    # ------------------------------------------------------------------

    def elbo_loss(
        self,
        x: torch.Tensor,
        pert_idx: torch.Tensor,
        control_mean: torch.Tensor,
        pert_mean: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute ELBO with consistency and proportion terms.

        Parameters
        ----------
        x:            (B, n_genes) observed expression.
        pert_idx:     (B,) integer perturbation index.
        control_mean: (n_genes,) training mean of control cells.
        pert_mean:    (B, n_genes) per-cell target mean (ground-truth mean
                      for that cell's perturbation, looked up at the batch
                      level).
        """
        out = self.forward(x, pert_idx)

        # Reconstruction (MSE)
        recon_loss = F.mse_loss(out["x_recon"], x, reduction="mean")

        # KL divergence  KL(N(μ,σ²) || N(0,1))
        kl_loss = -0.5 * torch.mean(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()
        )

        # Consistency: mean of decoded outputs should match target pert mean
        decoded_mean = out["x_recon"].mean(dim=0)
        true_shift = pert_mean.mean(dim=0) - control_mean
        pred_shift = decoded_mean - control_mean
        consistency_loss = F.mse_loss(pred_shift, true_shift)

        # Proportion entropy regularisation (encourage spreading)
        prop_entropy = -torch.mean(
            torch.sum(out["proportions"] * torch.log(out["proportions"] + 1e-8), dim=-1)
        )
        proportion_loss = -prop_entropy  # minimise → maximise entropy

        cfg = self.cfg
        total = (
            recon_loss
            + cfg.beta * kl_loss
            + cfg.gamma * consistency_loss
            + cfg.delta * proportion_loss
        )
        return {
            "loss": total,
            "recon": recon_loss,
            "kl": kl_loss,
            "consistency": consistency_loss,
            "proportion": proportion_loss,
        }


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class MeanShiftVAEPredictor:
    """High-level fit/predict/save/load wrapper for MeanShiftVAE.

    Parameters
    ----------
    cfg:
        Model and training hyperparameters.
    perturbation_col:
        ``adata.obs`` column with perturbation labels.
    control_label:
        Label for unperturbed / control cells.
    """

    def __init__(
        self,
        cfg: Optional[VAEConfig] = None,
        perturbation_col: str = "perturbation",
        control_label: str = "control",
    ) -> None:
        self.cfg = cfg or VAEConfig()
        self.perturbation_col = perturbation_col
        self.control_label = control_label

        self._model: Optional[MeanShiftVAE] = None
        self._pert_to_idx: Optional[Dict[str, int]] = None
        self._control_mean: Optional[np.ndarray] = None
        self._pert_means: Optional[np.ndarray] = None  # (n_perts, n_genes)
        self.gene_names_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        return torch.device(self.cfg.device)

    def _encode_labels(
        self, labels: np.ndarray
    ) -> Tuple[np.ndarray, list[str]]:
        """Map string labels → integer indices."""
        classes = list(self._pert_to_idx.keys())
        idx = np.array(
            [self._pert_to_idx.get(lbl, 0) for lbl in labels], dtype=np.int64
        )
        return idx, classes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata: ad.AnnData) -> "MeanShiftVAEPredictor":
        """Train MeanShiftVAE on *adata*.

        Parameters
        ----------
        adata:
            Training AnnData (cells × genes).

        Returns
        -------
        self
        """
        device = self._device()

        # Build label encoder
        labels = adata.obs[self.perturbation_col].values
        unique_perts = sorted(set(labels))
        self._pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
        n_perturbations = len(unique_perts)
        n_genes = adata.n_vars
        self.gene_names_ = np.asarray(adata.var_names)

        # Update config
        self.cfg.n_genes = n_genes
        self.cfg.n_perturbations = n_perturbations

        # Precompute per-perturbation means (used in consistency loss)
        means_df = compute_perturbation_means(adata, self.perturbation_col)
        self._pert_means = means_df.reindex(unique_perts).values.astype(np.float32)
        if self.control_label in means_df.index:
            self._control_mean = means_df.loc[self.control_label].values.astype(np.float32)
        else:
            self._control_mean = _dense(adata.X).mean(axis=0).astype(np.float32)

        control_mean_t = torch.tensor(self._control_mean, device=device)
        pert_means_t = torch.tensor(self._pert_means, device=device)

        # Build model
        self._model = MeanShiftVAE(self.cfg).to(device)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.cfg.lr)

        # DataLoader
        X_np = _dense(adata.X)
        pert_idx_np = np.array(
            [self._pert_to_idx[lbl] for lbl in labels], dtype=np.int64
        )
        dataset = TensorDataset(
            torch.tensor(X_np),
            torch.tensor(pert_idx_np),
        )
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False
        )

        self._model.train()
        logger.info(
            "Training MeanShiftVAE: %d epochs, %d cells, device=%s",
            self.cfg.n_epochs,
            adata.n_obs,
            device,
        )
        for epoch in tqdm(range(self.cfg.n_epochs), desc="VAE training", unit="epoch"):
            epoch_loss = 0.0
            for x_batch, pidx_batch in loader:
                x_batch = x_batch.to(device)
                pidx_batch = pidx_batch.to(device)
                # Gather per-sample target means
                pm_batch = pert_means_t[pidx_batch]  # (B, n_genes)

                losses = self._model.elbo_loss(
                    x_batch, pidx_batch, control_mean_t, pm_batch
                )
                optimiser.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optimiser.step()
                epoch_loss += losses["loss"].item()

            if (epoch + 1) % max(1, self.cfg.n_epochs // 10) == 0:
                logger.info(
                    "Epoch %d/%d  loss=%.4f",
                    epoch + 1,
                    self.cfg.n_epochs,
                    epoch_loss / len(loader),
                )

        self._model.eval()
        return self

    def predict(self, adata: ad.AnnData) -> np.ndarray:
        """Return predicted expression for each cell.

        Parameters
        ----------
        adata:
            AnnData with perturbation labels in ``obs[perturbation_col]``.

        Returns
        -------
        Array of shape ``(n_cells, n_genes)`` with float32 predictions.
        """
        if self._model is None or self._pert_to_idx is None:
            raise RuntimeError("Call fit() before predict().")

        device = self._device()
        labels = adata.obs[self.perturbation_col].values
        pert_idx_np = np.array(
            [self._pert_to_idx.get(lbl, 0) for lbl in labels], dtype=np.int64
        )
        X_np = _dense(adata.X)

        dataset = TensorDataset(
            torch.tensor(X_np),
            torch.tensor(pert_idx_np),
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)

        preds: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for x_batch, pidx_batch in loader:
                x_batch = x_batch.to(device)
                pidx_batch = pidx_batch.to(device)
                out = self._model(x_batch, pidx_batch)
                preds.append(out["x_recon"].cpu().numpy())

        return np.concatenate(preds, axis=0).astype(np.float32)

    def predict_proportions(self, adata: ad.AnnData) -> np.ndarray:
        """Return gene-program proportions for each cell.

        Returns
        -------
        Array of shape ``(n_cells, n_programs)`` with float32 proportions.
        """
        if self._model is None or self._pert_to_idx is None:
            raise RuntimeError("Call fit() before predict_proportions().")

        device = self._device()
        labels = adata.obs[self.perturbation_col].values
        pert_idx_np = np.array(
            [self._pert_to_idx.get(lbl, 0) for lbl in labels], dtype=np.int64
        )
        X_np = _dense(adata.X)

        dataset = TensorDataset(
            torch.tensor(X_np),
            torch.tensor(pert_idx_np),
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)

        props: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for x_batch, pidx_batch in loader:
                x_batch = x_batch.to(device)
                pidx_batch = pidx_batch.to(device)
                out = self._model(x_batch, pidx_batch)
                props.append(out["proportions"].cpu().numpy())

        return np.concatenate(props, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights and metadata to *path*.

        Saves a dict with the config, label mapping, means, and state dict.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("No model to save; call fit() first.")
        payload = {
            "cfg": self.cfg,
            "perturbation_col": self.perturbation_col,
            "control_label": self.control_label,
            "pert_to_idx": self._pert_to_idx,
            "control_mean": self._control_mean,
            "pert_means": self._pert_means,
            "gene_names": self.gene_names_,
            "state_dict": self._model.state_dict(),
        }
        torch.save(payload, path)
        logger.info("Saved MeanShiftVAEPredictor to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MeanShiftVAEPredictor":
        """Load from *path*."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg: VAEConfig = payload["cfg"]
        obj = cls(cfg=cfg)
        obj.perturbation_col = payload["perturbation_col"]
        obj.control_label = payload["control_label"]
        obj._pert_to_idx = payload["pert_to_idx"]
        obj._control_mean = payload["control_mean"]
        obj._pert_means = payload["pert_means"]
        obj.gene_names_ = payload["gene_names"]
        obj._model = MeanShiftVAE(cfg)
        obj._model.load_state_dict(payload["state_dict"])
        obj._model.eval()
        logger.info("Loaded MeanShiftVAEPredictor from %s", path)
        return obj
