"""Channel contribution and budget reallocation analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM


def channel_sensitivity(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute posterior mean contribution per channel.

    Returns DataFrame indexed by channel name with columns:
        mean_contribution, std_contribution, pct_contribution
    """
    # shape: (n_chains, n_draws, n_dates, n_channels) — stored as pm.Deterministic during fit
    contributions = mmm.idata.posterior["channel_contribution"].values
    # Flatten chain/draw/date dims; average over all but last (channel) axis
    flat_axes = tuple(range(contributions.ndim - 1))
    mean_c = contributions.mean(axis=flat_axes)
    std_c = contributions.std(axis=flat_axes)
    total = mean_c.sum()
    pct_c = mean_c / total * 100 if total > 0 else np.zeros_like(mean_c)
    return pd.DataFrame(
        {"mean_contribution": mean_c, "std_contribution": std_c, "pct_contribution": pct_c},
        index=mmm.channel_columns,
    )


def budget_reallocation(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare actual vs contribution-implied optimal budget allocation.

    Returns DataFrame with columns:
        actual_spend, actual_pct, optimal_pct, reallocation_delta_pct
    """
    actual_spend = df[mmm.channel_columns].sum()
    total_spend = actual_spend.sum()
    actual_pct = actual_spend / total_spend * 100 if total_spend > 0 else actual_spend * 0

    sensitivity = channel_sensitivity(mmm, df)
    optimal_pct = sensitivity["pct_contribution"]

    return pd.DataFrame({
        "actual_spend": actual_spend,
        "actual_pct": actual_pct,
        "optimal_pct": optimal_pct,
        "reallocation_delta_pct": optimal_pct - actual_pct,
    })
