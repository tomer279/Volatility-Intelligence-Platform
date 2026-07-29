# How to Add a Feature

This tutorial walks through adding a new feature to the Volatility Intelligence
Platform, from implementation to verifying its importance ranking.

---

## Step 1 — Decide the Feature Family

Features are grouped by family:

| Family | Examples | Module |
|--------|----------|--------|
| HAR (trailing RV) | `rv_cc_1d`, `rv_cc_5d`, `rv_cc_21d` | `features.trailing_rv` |
| Returns | `ret_1d`, `ret_5d` | `features.targets` |
| Range | `range_1d`, `range_5d_mean` | `features.targets` |
| Volume | `volume_z_21d` | `features.targets` |
| Cross-asset | `vix_level`, `vix_chg_1d` | `features.cross_asset` |

Pick the family that best describes your new feature.  If it does not fit an
existing family, consider whether it warrants a new module under
`src/vip/features/`.

---

## Step 2 — Implement the Computation Function

Write a pure function that takes a canonical OHLCV DataFrame and returns a
Series (or single-column DataFrame).  The function must use only data up to and
including row *t* — no future information.

**Example: 10-day trailing realized volatility**

```python
# src/vip/features/trailing_rv.py  (add to existing module)

import numpy as np
import pandas as pd


def realized_volatility_trailing_10d(ohlcv: pd.DataFrame) -> pd.Series:
    """Compute 10-day trailing close-to-close realized volatility.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Canonical OHLCV DataFrame with a ``close`` column.

    Returns
    -------
    pd.Series
        Trailing 10-day RV, indexed to match ``ohlcv``.
    """
    log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    return log_ret.rolling(window=10).apply(
        lambda x: np.sqrt((x ** 2).sum()), raw=True
    ).rename("rv_cc_10d")
```

Key rules:

- Input is the canonical OHLCV frame (lowercase columns, DatetimeIndex).
- Output Series name becomes the column name in the feature matrix.
- Use only backward-looking windows (`shift`, `rolling`).  Never reference
  future rows.

---

## Step 3 — Register in the Feature Pipeline

`src/vip/features/pipeline.py` builds the final feature matrix by calling the feature registry to materialize the requested feature names.
So your feature must be registered in the default `FeatureRegistry` that the pipeline uses (typically via `create_default_registry()`).

Open `src/vip/features/registry.py` and add a `FeatureSpec` inside `create_default_registry()`:

```python
from vip.features.trailing_rv import realized_volatility_trailing_10d

# Inside create_default_registry():
registry.register(
    FeatureSpec(
        name="rv_cc_10d",
        builder=realized_volatility_trailing_10d,
        family="har",
    )
)
```

After registration, `build_feature_matrix` will automatically include
`rv_cc_10d` in every feature build.

---

## Step 4 — Write a Leakage Test

Add a test that asserts the feature at date *t* uses only data up to *t*.  The
standard pattern: mask future data with NaN and verify the feature value does not
change.

```python
# tests/unit/features/test_rv_cc_10d_leakage.py

import numpy as np
import pandas as pd
from vip.features.trailing_rv import realized_volatility_trailing_10d


def test_rv_cc_10d_no_future_leakage():
    """Feature at t must not change when data after t is modified."""
    dates = pd.bdate_range("2023-01-01", periods=30)
    rng = np.random.default_rng(42)
    ohlcv = pd.DataFrame(
        {
            "open": 100 + rng.standard_normal(30).cumsum(),
            "high": 102 + rng.standard_normal(30).cumsum(),
            "low": 98 + rng.standard_normal(30).cumsum(),
            "close": 100 + rng.standard_normal(30).cumsum(),
            "volume": rng.integers(1_000, 10_000, size=30),
        },
        index=dates,
    )

    full = realized_volatility_trailing_10d(ohlcv)

    check_idx = 19  # date t = dates[19]
    truncated = ohlcv.iloc[: check_idx + 1].copy()
    partial = realized_volatility_trailing_10d(truncated)

    assert np.isclose(full.iloc[check_idx], partial.iloc[-1], rtol=1e-12)
```

Run it:

```powershell
py -m pytest tests/unit/features/test_rv_cc_10d_leakage.py -q
```

---

## Step 5 — Rebuild the Feature Matrix

```powershell
vip features --symbol SPY
```

This re-runs the full feature pipeline.  Verify the new column appears:

```powershell
py -c "import pandas as pd; df = pd.read_parquet('data/features/SPY.parquet'); print(df.columns.tolist())"
```

You should see `rv_cc_10d` in the column list.

---

## Step 6 — Run the Screen to See Importance

```powershell
vip screen --symbol SPY
```

Open `data/artifacts/factor-screen-SPY-<date>/report.html` and check the
importance ranking table.  The new feature will appear with its median ΔQLIKE
importance and SHAP rank (if `--with-shap` is used).

To include VIX cross-asset features:

```powershell
vip screen --symbol SPY --with-vix
```

---

## Summary Checklist

- [ ] Implemented a backward-looking computation function
- [ ] Registered a `FeatureSpec` in `create_default_registry()`
- [ ] Wrote a leakage test (feature at *t* unchanged by future data)
- [ ] Rebuilt the feature matrix with `vip features`
- [ ] Ran `vip screen` and verified the feature appears in the ranking

