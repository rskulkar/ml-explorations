"""Unit tests for evaluation/metrics.py using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from broad_obesity.evaluation.metrics import (
    pearson_delta,
    mmd_score,
    l1_distance,
    combined_score,
    _MMD_BANDWIDTHS,
)

RNG = np.random.default_rng(0)
N_GENES = 50
N_CELLS = 80


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def control_cells() -> np.ndarray:
    return RNG.normal(0, 1, size=(N_CELLS, N_GENES)).astype(np.float32)


@pytest.fixture()
def true_perturbed(control_cells) -> np.ndarray:
    """A shifted version of control: clear ground-truth signal."""
    shift = RNG.normal(2, 0.5, size=N_GENES).astype(np.float32)
    return control_cells + shift


@pytest.fixture()
def good_pred(true_perturbed) -> np.ndarray:
    """Predictions very close to the ground truth."""
    noise = RNG.normal(0, 0.05, size=true_perturbed.shape).astype(np.float32)
    return true_perturbed + noise


@pytest.fixture()
def bad_pred(control_cells) -> np.ndarray:
    """Predictions equal to control (worst case)."""
    return control_cells.copy()


@pytest.fixture()
def control_mean(control_cells) -> np.ndarray:
    return control_cells.mean(axis=0)


# ---------------------------------------------------------------------------
# pearson_delta
# ---------------------------------------------------------------------------

class TestPearsonDelta:
    def test_perfect_prediction(self, true_perturbed, control_mean):
        r = pearson_delta(true_perturbed, true_perturbed, control_mean)
        assert pytest.approx(r, abs=1e-5) == 1.0

    def test_good_prediction_high_r(self, good_pred, true_perturbed, control_mean):
        r = pearson_delta(good_pred, true_perturbed, control_mean)
        assert r > 0.95

    def test_bad_prediction_low_r(self, bad_pred, true_perturbed, control_mean):
        # Predicting control when truth is shifted → low correlation
        r = pearson_delta(bad_pred, true_perturbed, control_mean)
        assert r < 0.5

    def test_range(self, good_pred, true_perturbed, control_mean):
        r = pearson_delta(good_pred, true_perturbed, control_mean)
        assert -1.0 <= r <= 1.0

    def test_constant_delta_returns_zero(self):
        # Use exact zero control_mean so pred_delta and true_delta are
        # identically the zero vector, triggering the std==0 guard.
        ctrl = np.zeros(N_GENES, dtype=np.float32)
        zeros = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
        r = pearson_delta(zeros, zeros, ctrl)
        assert r == 0.0


# ---------------------------------------------------------------------------
# mmd_score
# ---------------------------------------------------------------------------

class TestMMDScore:
    def test_identical_distributions_near_zero(self, true_perturbed):
        score = mmd_score(true_perturbed, true_perturbed)
        assert score >= 0.0
        assert score < 1e-3

    def test_different_distributions_positive(self, true_perturbed, bad_pred):
        score = mmd_score(true_perturbed, bad_pred)
        assert score > 0.0

    def test_good_pred_lower_mmd(self, good_pred, bad_pred, true_perturbed):
        mmd_good = mmd_score(good_pred, true_perturbed)
        mmd_bad = mmd_score(bad_pred, true_perturbed)
        assert mmd_good < mmd_bad

    def test_non_negative(self, good_pred, true_perturbed):
        score = mmd_score(good_pred, true_perturbed)
        assert score >= 0.0

    def test_default_bandwidths_used(self, true_perturbed):
        s1 = mmd_score(true_perturbed, true_perturbed, bandwidths=_MMD_BANDWIDTHS)
        assert s1 >= 0.0

    def test_small_sample(self):
        tiny = RNG.normal(size=(3, N_GENES)).astype(np.float32)
        score = mmd_score(tiny, tiny)
        assert score >= 0.0

    def test_fewer_than_two_cells_returns_zero(self):
        single = RNG.normal(size=(1, N_GENES)).astype(np.float32)
        score = mmd_score(single, single)
        assert score == 0.0


# ---------------------------------------------------------------------------
# l1_distance
# ---------------------------------------------------------------------------

class TestL1Distance:
    def test_identical_returns_zero(self, true_perturbed):
        d = l1_distance(true_perturbed, true_perturbed)
        assert pytest.approx(d, abs=1e-6) == 0.0

    def test_non_negative(self, good_pred, true_perturbed):
        assert l1_distance(good_pred, true_perturbed) >= 0.0

    def test_good_pred_lower_l1(self, good_pred, bad_pred, true_perturbed):
        l1_good = l1_distance(good_pred, true_perturbed)
        l1_bad = l1_distance(bad_pred, true_perturbed)
        assert l1_good < l1_bad

    def test_known_value(self):
        a = np.ones((10, 4), dtype=np.float32)
        b = np.zeros((10, 4), dtype=np.float32)
        assert pytest.approx(l1_distance(a, b), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# combined_score
# ---------------------------------------------------------------------------

class TestCombinedScore:
    def test_returns_expected_keys(self, good_pred, true_perturbed, control_cells):
        result = combined_score(good_pred, true_perturbed, control_cells)
        expected_keys = {
            "score", "S_R", "S_L",
            "pearson_r", "mmd_r", "l1_r",
            "pearson_l", "mmd_l", "l1_l",
        }
        assert set(result.keys()) == expected_keys

    def test_good_pred_beats_bad_pred(
        self, good_pred, bad_pred, true_perturbed, control_cells
    ):
        s_good = combined_score(good_pred, true_perturbed, control_cells)["score"]
        s_bad = combined_score(bad_pred, true_perturbed, control_cells)["score"]
        assert s_good > s_bad

    def test_weights_sum_formula(self, good_pred, true_perturbed, control_cells):
        result = combined_score(
            good_pred, true_perturbed, control_cells, w_real=0.75, w_ctrl=0.25
        )
        expected = 0.75 * result["S_R"] + 0.25 * result["S_L"]
        assert pytest.approx(result["score"], rel=1e-5) == expected
