"""Unit tests for mean_shift_vae.py using synthetic data (no real dataset)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import anndata as ad

from broad_obesity.models.mean_shift_vae import (
    VAEConfig,
    Encoder,
    Decoder,
    GeneEmbedding,
    ProgramProportionHead,
    MeanShiftVAE,
    MeanShiftVAEPredictor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_GENES = 40
N_CELLS = 120
N_PERTS = 5
N_LATENT = 16
N_HIDDEN = 32
N_PROGRAMS = 4
PERT_EMB = 8
BATCH = 16


def _make_cfg() -> VAEConfig:
    return VAEConfig(
        n_genes=N_GENES,
        n_perturbations=N_PERTS,
        latent_dim=N_LATENT,
        hidden_dim=N_HIDDEN,
        n_programs=N_PROGRAMS,
        pert_emb_dim=PERT_EMB,
        beta=1e-3,
        gamma=0.1,
        delta=0.01,
        n_epochs=3,
        batch_size=32,
        device="cpu",
    )


def _make_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    pert_labels = ["control"] + [f"gene_{i}" for i in range(N_PERTS - 1)]
    labels = rng.choice(pert_labels, size=N_CELLS)
    X = rng.exponential(scale=2.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = {"perturbation": labels}
    adata = ad.AnnData(
        X=X,
        obs={k: list(v) for k, v in obs.items()},
    )
    adata.var_names = [f"gene_{i}" for i in range(N_GENES)]
    return adata


# ---------------------------------------------------------------------------
# Sub-module shapes
# ---------------------------------------------------------------------------

class TestGeneEmbedding:
    def test_output_shape(self):
        ge = GeneEmbedding(N_GENES, PERT_EMB)
        out = ge()
        assert out.shape == (N_GENES, PERT_EMB)

    def test_grad_flows(self):
        ge = GeneEmbedding(N_GENES, PERT_EMB)
        out = ge()
        loss = out.sum()
        loss.backward()
        assert ge.embedding.weight.grad is not None


class TestEncoder:
    def test_output_shapes(self):
        cfg = _make_cfg()
        enc = Encoder(cfg)
        x = torch.randn(BATCH, N_GENES)
        p = torch.randn(BATCH, PERT_EMB)
        mu, lv = enc(x, p)
        assert mu.shape == (BATCH, N_LATENT)
        assert lv.shape == (BATCH, N_LATENT)

    def test_different_inputs_give_different_outputs(self):
        cfg = _make_cfg()
        enc = Encoder(cfg)
        x1 = torch.randn(BATCH, N_GENES)
        x2 = torch.randn(BATCH, N_GENES)
        p = torch.randn(BATCH, PERT_EMB)
        mu1, _ = enc(x1, p)
        mu2, _ = enc(x2, p)
        assert not torch.allclose(mu1, mu2)


class TestDecoder:
    def test_output_shape(self):
        cfg = _make_cfg()
        dec = Decoder(cfg)
        z = torch.randn(BATCH, N_LATENT)
        p = torch.randn(BATCH, PERT_EMB)
        out = dec(z, p)
        assert out.shape == (BATCH, N_GENES)


class TestProgramProportionHead:
    def test_output_shape(self):
        head = ProgramProportionHead(N_LATENT, N_PROGRAMS)
        z = torch.randn(BATCH, N_LATENT)
        props = head(z)
        assert props.shape == (BATCH, N_PROGRAMS)

    def test_sums_to_one(self):
        head = ProgramProportionHead(N_LATENT, N_PROGRAMS)
        z = torch.randn(BATCH, N_LATENT)
        props = head(z)
        row_sums = props.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(BATCH), atol=1e-5)

    def test_all_non_negative(self):
        head = ProgramProportionHead(N_LATENT, N_PROGRAMS)
        z = torch.randn(BATCH, N_LATENT)
        props = head(z)
        assert (props >= 0).all()


# ---------------------------------------------------------------------------
# MeanShiftVAE forward & loss
# ---------------------------------------------------------------------------

class TestMeanShiftVAE:
    def setup_method(self):
        self.cfg = _make_cfg()
        self.model = MeanShiftVAE(self.cfg)

    def _batch(self):
        x = torch.randn(BATCH, N_GENES)
        pidx = torch.randint(0, N_PERTS, (BATCH,))
        return x, pidx

    def test_forward_output_keys(self):
        x, pidx = self._batch()
        out = self.model(x, pidx)
        assert set(out.keys()) == {"x_recon", "mu", "logvar", "z", "proportions"}

    def test_forward_shapes(self):
        x, pidx = self._batch()
        out = self.model(x, pidx)
        assert out["x_recon"].shape == (BATCH, N_GENES)
        assert out["mu"].shape == (BATCH, N_LATENT)
        assert out["logvar"].shape == (BATCH, N_LATENT)
        assert out["z"].shape == (BATCH, N_LATENT)
        assert out["proportions"].shape == (BATCH, N_PROGRAMS)

    def test_elbo_loss_returns_scalar(self):
        x, pidx = self._batch()
        ctrl_mean = torch.zeros(N_GENES)
        pert_mean = torch.zeros(BATCH, N_GENES)
        losses = self.model.elbo_loss(x, pidx, ctrl_mean, pert_mean)
        assert losses["loss"].shape == ()

    def test_elbo_loss_components_finite(self):
        x, pidx = self._batch()
        ctrl_mean = torch.zeros(N_GENES)
        pert_mean = torch.zeros(BATCH, N_GENES)
        losses = self.model.elbo_loss(x, pidx, ctrl_mean, pert_mean)
        # Reconstruction loss is always non-negative
        assert losses["recon"].item() >= 0
        # proportion_loss = -entropy, which can be negative (maximising entropy)
        assert np.isfinite(losses["proportion"].item())
        assert np.isfinite(losses["loss"].item())

    def test_reparameterise_stochastic(self):
        mu = torch.zeros(BATCH, N_LATENT)
        logvar = torch.zeros(BATCH, N_LATENT)  # σ=1
        z1 = MeanShiftVAE.reparameterise(mu, logvar)
        z2 = MeanShiftVAE.reparameterise(mu, logvar)
        assert not torch.allclose(z1, z2)

    def test_gradients_flow(self):
        x, pidx = self._batch()
        ctrl_mean = torch.zeros(N_GENES)
        pert_mean = torch.zeros(BATCH, N_GENES)
        losses = self.model.elbo_loss(x, pidx, ctrl_mean, pert_mean)
        losses["loss"].backward()
        grads = [
            p.grad for p in self.model.parameters() if p.grad is not None
        ]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# MeanShiftVAEPredictor fit / predict / save / load
# ---------------------------------------------------------------------------

class TestMeanShiftVAEPredictor:
    def _predictor(self) -> MeanShiftVAEPredictor:
        cfg = _make_cfg()
        return MeanShiftVAEPredictor(
            cfg=cfg,
            perturbation_col="perturbation",
            control_label="control",
        )

    def test_fit_predict_shape(self, tmp_path):
        adata = _make_adata()
        pred = self._predictor()
        pred.fit(adata)
        preds = pred.predict(adata)
        assert preds.shape == (N_CELLS, N_GENES)
        assert preds.dtype == np.float32

    def test_predict_proportions_shape(self):
        adata = _make_adata()
        pred = self._predictor()
        pred.fit(adata)
        props = pred.predict_proportions(adata)
        assert props.shape == (N_CELLS, N_PROGRAMS)

    def test_proportions_sum_to_one(self):
        adata = _make_adata()
        pred = self._predictor()
        pred.fit(adata)
        props = pred.predict_proportions(adata)
        row_sums = props.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(N_CELLS), atol=1e-4)

    def test_save_load_roundtrip(self, tmp_path):
        adata = _make_adata()
        pred = self._predictor()
        pred.fit(adata)
        preds_before = pred.predict(adata)

        model_path = tmp_path / "vae.pt"
        pred.save(model_path)

        loaded = MeanShiftVAEPredictor.load(model_path)
        preds_after = loaded.predict(adata)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)

    def test_predict_before_fit_raises(self):
        pred = self._predictor()
        adata = _make_adata()
        with pytest.raises(RuntimeError, match="fit"):
            pred.predict(adata)

    def test_unseen_perturbation_does_not_crash(self):
        adata_train = _make_adata(seed=1)
        adata_test = _make_adata(seed=2)
        # Inject an unseen perturbation
        adata_test.obs["perturbation"] = "totally_new_gene"

        pred = self._predictor()
        pred.fit(adata_train)
        preds = pred.predict(adata_test)  # should not raise
        assert preds.shape == (N_CELLS, N_GENES)

    def test_gene_names_stored(self):
        adata = _make_adata()
        pred = self._predictor()
        pred.fit(adata)
        assert pred.gene_names_ is not None
        assert len(pred.gene_names_) == N_GENES
