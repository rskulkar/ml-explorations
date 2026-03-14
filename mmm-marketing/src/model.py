"""MMM model construction and fitting."""
from __future__ import annotations
import arviz as az
import pandas as pd
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

CHANNEL_COLUMNS = ["x1", "x2"]
CONTROL_COLUMNS = ["event_1", "event_2", "t"]
DATE_COLUMN = "date_week"
TARGET_COLUMN = "y"


def build_mmm(
    channel_columns: list[str] = CHANNEL_COLUMNS,
    control_columns: list[str] = CONTROL_COLUMNS,
    yearly_seasonality: int = 2,
) -> MMM:
    """Construct MMM with geometric adstock + logistic saturation + yearly seasonality."""
    return MMM(
        date_column=DATE_COLUMN,
        channel_columns=channel_columns,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        control_columns=control_columns,
        yearly_seasonality=yearly_seasonality,
    )


def fit_mmm(
    mmm: MMM,
    df: pd.DataFrame,
    tune: int = 500,
    draws: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Fit MMM via MCMC. cores=1 required for M1/MPS safety."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return mmm.fit(
        X, y,
        tune=tune, draws=draws, chains=chains,
        target_accept=target_accept, cores=1,
        random_seed=random_seed,
    )
