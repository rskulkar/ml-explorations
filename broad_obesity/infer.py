#!/usr/bin/env python
"""Inference entrypoint for the Broad Obesity competition.

Usage examples
--------------
  uv run python infer.py --baseline perturbed_mean \\
      --model models/perturbed_mean.pkl \\
      --data  data/test.h5ad \\
      --output submissions/

  uv run python infer.py --baseline vae \\
      --model models/mean_shift_vae.pt \\
      --data  data/test.h5ad \\
      --output submissions/ \\
      --write-proportions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run perturbation prediction inference.")
    p.add_argument(
        "--baseline",
        choices=["perturbed_mean", "linear", "vae"],
        required=True,
        help="Which model type to load.",
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to saved model file (pkl or pt).",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to test/inference AnnData h5ad.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("submissions"),
        help="Output directory for submission files.",
    )
    p.add_argument(
        "--perturbation-col",
        default="perturbation",
        help="obs column with perturbation labels.",
    )
    p.add_argument(
        "--write-proportions",
        action="store_true",
        help="Also write gene-program proportions (VAE only).",
    )
    p.add_argument(
        "--method-name",
        default=None,
        help="Method name for the submission description JSON.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from broad_obesity.data.loader import load_adata
    from broad_obesity.utils.submission import (
        write_prediction_h5ad,
        write_program_proportions,
        write_method_description,
    )

    logger.info("Loading inference data from %s", args.data)
    adata = load_adata(args.data)

    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if args.baseline == "perturbed_mean":
        from broad_obesity.models.perturbed_mean import PerturbedMeanPredictor

        model = PerturbedMeanPredictor.load(args.model)
        method_name = args.method_name or "PerturbedMeanPredictor"
        description = (
            "Predicts each cell's expression as the training mean for its "
            "perturbation (simple mean baseline)."
        )

    elif args.baseline == "linear":
        from broad_obesity.models.linear_baseline import LinearPerturbationPredictor

        model = LinearPerturbationPredictor.load(args.model)
        method_name = args.method_name or "LinearPerturbationPredictor"
        description = (
            "Ridge regression from one-hot perturbation indicator to gene "
            "expression (regularised mean baseline)."
        )

    elif args.baseline == "vae":
        from broad_obesity.models.mean_shift_vae import MeanShiftVAEPredictor

        model = MeanShiftVAEPredictor.load(args.model)
        method_name = args.method_name or "MeanShiftVAE"
        description = (
            "Variational autoencoder conditioned on perturbation identity; "
            "trained with ELBO + consistency + proportion losses."
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    logger.info("Running prediction on %d cells …", adata.n_obs)
    preds = model.predict(adata)

    pred_path = args.output / "predictions.h5ad"
    write_prediction_h5ad(preds, adata, pred_path)
    logger.info("Predictions written to %s", pred_path)

    # ------------------------------------------------------------------
    # Optional: program proportions (VAE only)
    # ------------------------------------------------------------------
    if args.write_proportions:
        if args.baseline != "vae":
            logger.warning(
                "--write-proportions is only supported for --baseline vae; skipping."
            )
        else:
            proportions = model.predict_proportions(adata)
            prop_path = args.output / "program_proportions.csv"
            write_program_proportions(proportions, adata.obs_names, prop_path)
            logger.info("Program proportions written to %s", prop_path)

    # ------------------------------------------------------------------
    # Method description
    # ------------------------------------------------------------------
    desc_path = args.output / "method_description.json"
    hparams = {}
    if hasattr(model, "cfg"):
        import dataclasses

        hparams = dataclasses.asdict(model.cfg)
    elif hasattr(model, "alpha"):
        hparams = {"alpha": model.alpha}

    write_method_description(desc_path, method_name, description, hparams)
    logger.info("Method description written to %s", desc_path)
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
