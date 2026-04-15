#!/usr/bin/env python
"""Training entrypoint for the Broad Obesity competition.

Usage examples
--------------
  uv run python train.py --baseline perturbed_mean --data data/train.h5ad
  uv run python train.py --baseline linear          --data data/train.h5ad
  uv run python train.py --baseline vae             --data data/train.h5ad \\
      --epochs 100 --latent-dim 128 --hidden-dim 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a perturbation prediction model."
    )
    p.add_argument(
        "--baseline",
        choices=["perturbed_mean", "linear", "vae"],
        default="perturbed_mean",
        help="Which model to train.",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training AnnData h5ad file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Directory in which to save the trained model.",
    )
    p.add_argument(
        "--perturbation-col",
        default="perturbation",
        help="obs column with perturbation labels.",
    )
    p.add_argument(
        "--control-label",
        default="control",
        help="Label for unperturbed cells.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of cells held out for validation.",
    )
    p.add_argument("--seed", type=int, default=42)
    # VAE-specific
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--latent-dim", type=int,   default=64)
    p.add_argument("--hidden-dim", type=int,   default=256)
    p.add_argument("--n-programs", type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--beta",       type=float, default=1e-3,  help="KL weight")
    p.add_argument("--gamma",      type=float, default=0.1,   help="Consistency weight")
    p.add_argument("--device",     default="cpu")
    # Ridge-specific
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge α")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from broad_obesity.data.loader import load_adata, train_val_split
    from broad_obesity.evaluation.metrics import combined_score

    logger.info("Loading data from %s", args.data)
    adata = load_adata(args.data)

    logger.info("Splitting train/val (val_frac=%.2f)", args.val_frac)
    train_adata, val_adata = train_val_split(
        adata,
        val_frac=args.val_frac,
        perturbation_col=args.perturbation_col,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build and train
    # ------------------------------------------------------------------
    if args.baseline == "perturbed_mean":
        from broad_obesity.models.perturbed_mean import PerturbedMeanPredictor

        model = PerturbedMeanPredictor(
            perturbation_col=args.perturbation_col,
            control_label=args.control_label,
        )
        model.fit(train_adata)
        save_path = args.output / "perturbed_mean.pkl"
        model.save(save_path)

    elif args.baseline == "linear":
        from broad_obesity.models.linear_baseline import LinearPerturbationPredictor

        model = LinearPerturbationPredictor(
            perturbation_col=args.perturbation_col,
            alpha=args.alpha,
        )
        model.fit(train_adata)
        save_path = args.output / "linear_baseline.pkl"
        model.save(save_path)

    elif args.baseline == "vae":
        from broad_obesity.models.mean_shift_vae import MeanShiftVAEPredictor, VAEConfig
        import torch

        device = args.device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        cfg = VAEConfig(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            n_programs=args.n_programs,
            lr=args.lr,
            beta=args.beta,
            gamma=args.gamma,
            device=device,
        )
        model = MeanShiftVAEPredictor(
            cfg=cfg,
            perturbation_col=args.perturbation_col,
            control_label=args.control_label,
        )
        model.fit(train_adata)
        save_path = args.output / "mean_shift_vae.pt"
        model.save(save_path)

    logger.info("Model saved to %s", save_path)

    # ------------------------------------------------------------------
    # Quick validation
    # ------------------------------------------------------------------
    logger.info("Running validation …")
    from broad_obesity.data.loader import _dense

    preds = model.predict(val_adata)
    control_mask = (
        val_adata.obs[args.perturbation_col] == args.control_label
    ).values

    if control_mask.sum() < 2:
        logger.warning("Too few control cells in val set; skipping combined_score.")
        return

    import numpy as np

    control_cells = _dense(val_adata.X)[control_mask]
    pert_mask = ~control_mask
    if pert_mask.sum() < 2:
        logger.warning("Too few perturbed cells in val set; skipping combined_score.")
        return

    true_pert = _dense(val_adata.X)[pert_mask]
    pred_pert = preds[pert_mask]

    scores = combined_score(pred_pert, true_pert, control_cells)
    logger.info(
        "Validation — score=%.4f  S_R=%.4f  S_L=%.4f  "
        "pearson_r=%.4f  mmd_r=%.4f  l1_r=%.4f",
        scores["score"],
        scores["S_R"],
        scores["S_L"],
        scores["pearson_r"],
        scores["mmd_r"],
        scores["l1_r"],
    )


if __name__ == "__main__":
    main()
