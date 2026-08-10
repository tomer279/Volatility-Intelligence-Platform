"""CLI helpers for optional feature extras.

Exports
-------
parse_feature_extras
    Parse ``--with`` tokens into ``FeatureMatrixExtras``.
"""

from __future__ import annotations

import typer

from vip.application.build_feature_matrix import FeatureMatrixExtras

_ALLOWED_TOKENS = frozenset({"vix", "jump", "iv_rv", "rates"})


def parse_feature_extras(raw: str) -> FeatureMatrixExtras:
    """Parse ``--with`` tokens into ``FeatureMatrixExtras``.

    Parameters
    ----------
    raw : str
        Comma-separated tokens (``vix``, ``jump``, ``iv_rv``, ``rates``).
        Empty means none. Token ``iv_rv`` implies VIX load
        (``include_vix=True``). Token ``rates`` loads TNX yield features.

    Returns
    -------
    FeatureMatrixExtras
        Parsed feature-build options.

    Raises
    ------
    typer.BadParameter
        If an unknown token is present.
    """
    tokens = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = tokens - _ALLOWED_TOKENS
    if unknown:
        bad = ", ".join(sorted(unknown))
        allowed = ", ".join(sorted(_ALLOWED_TOKENS))
        raise typer.BadParameter(
            f"Unknown --with token(s): {bad}. Allowed: {allowed}."
        )
    include_iv_rv = "iv_rv" in tokens
    return FeatureMatrixExtras(
        include_vix=("vix" in tokens) or include_iv_rv,
        include_jump="jump" in tokens,
        include_iv_rv=include_iv_rv,
        include_rates="rates" in tokens,
    )
