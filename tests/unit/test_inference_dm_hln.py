"""Tests for Newey–West lag helper and HLN–DM path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vip.evaluation.inference import (
    hln_diebold_mariano,
    nw_lags_for_horizon,
)


def test_nw_lags_for_five_day_horizon() -> None:
    """Locked research default: horizon 5 → NW lags 4."""
    assert nw_lags_for_horizon(5) == 4


def test_hln_differs_from_raw_dm_on_small_t() -> None:
    """HLN correction should change the statistic on small samples."""
    rng = np.random.default_rng(3)
    differential = pd.Series(rng.normal(-0.1, 1.0, size=40))
    nw_lags = nw_lags_for_horizon(5)
    result = hln_diebold_mariano(differential, nw_lags=nw_lags, horizon_days=5)
    assert result.nw_lags == 4
    assert result.hln_stat != pytest.approx(result.dm_stat)
    assert 0.0 < result.hln_pvalue < 1.0