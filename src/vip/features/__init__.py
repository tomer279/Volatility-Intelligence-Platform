"""Feature engineering and realized-volatility targets.

Exports
-------
build_target_rv_cc
    Build a forward close-to-close realized-volatility target series.
daily_log_returns
    Compute daily log returns from a close-price series.
realized_variance_forward
    Compute forward realized variance over a trading-day horizon.
realized_volatility_forward
    Compute forward realized volatility over a trading-day horizon.
realized_variance_trailing
    Compute trailing realized variance ending at session ``t``.
realized_volatility_trailing
    Compute trailing realized volatility ending at session ``t``.
bipower_variation_trailing
    Trailing daily bipower-variation proxy ending at session ``t``.
bipower_volatility_trailing
    Square root of trailing daily bipower variation.
jump_proportion_trailing
    Trailing jump-proportion proxy ``max(0, RV - BPV) / RV``.
build_jump_features
    Build jump-proportion columns at HAR windows (BPV used internally only).
FeatureSpec
    Metadata and builder callable for one feature family.
FeatureRegistry
    Register builders and assemble selected feature columns.
create_default_registry
    Build a registry with Milestone 2 families; optional ``jump`` via
    ``include_jump=True``.
build_feature_matrix
    Build features and target, then drop incomplete rows.
"""

from vip.features.pipeline import build_feature_matrix
from vip.features.jump_features import build_jump_features
from vip.features.realized import (
    realized_variance_trailing,
    realized_volatility_trailing,
    bipower_variation_trailing,
    bipower_volatility_trailing,
    jump_proportion_trailing,
)
from vip.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    create_default_registry,
)
from vip.features.targets import (
    build_target_rv_cc,
    daily_log_returns,
    realized_variance_forward,
    realized_volatility_forward,
)

__all__ = [
    "FeatureRegistry",
    "FeatureSpec",
    "build_feature_matrix",
    "build_target_rv_cc",
    "create_default_registry",
    "daily_log_returns",
    "realized_variance_forward",
    "realized_variance_trailing",
    "realized_volatility_forward",
    "realized_volatility_trailing",
    "build_jump_features",
    "bipower_variation_trailing",
    "bipower_volatility_trailing",
    "jump_proportion_trailing",
]