"""CLI helpers for optional feature extras.

Exports
-------
parse_feature_extras
    Parse ``--with`` tokens into ``FeatureMatrixExtras``.
"""

from __future__ import annotations

import typer

from vip.application.build_feature_matrix import FeatureMatrixExtras

_ALLOWED_TOKENS = frozenset({"vix", "jump"})


def parse_feature_extras(raw: str) -> FeatureMatrixExtras:
    """Parse ``--with`` tokens into ``FeatureMatrixExtras``.

    Parameters
    ----------
    raw : str
        Comma-separated tokens (``vix``, ``jump``). Empty means neither.

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
    return FeatureMatrixExtras(
        include_vix="vix" in tokens,
        include_jump="jump" in tokens,
    )