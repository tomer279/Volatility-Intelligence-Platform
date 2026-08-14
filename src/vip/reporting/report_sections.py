"""Thematic factor-screen memo sections (Implied, Parametric).

Exports
-------
ImpliedVsRealizedSection
    Context for the HTML “Implied vs realized” memo block.
ParametricVsHarSection
    Context for the HTML “Parametric vs HAR” memo block.
ReportExtras
    Non-tabular extras (image, caveats, Implied / Parametric sections).
build_implied_vs_realized_section
    Derive IV-model row + optional gap-feature rows from screen tables.
build_parametric_vs_har_section
    Derive OU (and optional stretch filter) rows from horse-race tables.
implied_section_template_payload
    Flatten Implied section inputs for Jinja2.
parametric_section_template_payload
    Flatten Parametric section inputs for Jinja2.
"""

from __future__ import annotations

from dataclasses import dataclass

VIX_AS_FORECAST_MODEL = "vix_as_forecast"
IV_RV_GAP_FEATURE_PREFIX = "vix_minus_rv_"
DEFAULT_TOP_IV_RV_GAP_FEATURES = 3

VIX_PROXY_CAVEAT = (
    "VIX is used as an IV proxy for index/ETF research; it is not "
    "single-name implied volatility and is not a variance-swap or "
    "options-replication identity."
)
VIX_UNIT_CONVERSION_NOTE = (
    "Unit conversion (locked): "
    "vix_vol_daily = (vix_level / 100.0) / sqrt(252), "
    "putting VIX on the same non-annualized daily-vol scale as rv_cc_*."
)

OU_RV_MODEL = "ou_rv"
STRETCH_FILTER_MODEL = "ewma_recursive"
PARAMETRIC_VS_HAR_CAVEATS = (
    "Discrete OU / AR(1) on log realized-vol target; not continuous-time SV "
    "or Heston.",
    "State is the log of the training target; the h-step conditional mean is "
    "exp-mapped and floored.",
    "Core MVP freezes the end-of-train log state (same spirit as frozen EWMA); "
    "it does not recurse on test labels.",
    "Say 'significantly better' only via the locked bootstrap wording "
    "(reject at α and mean ΔQLIKE < 0); point QLIKE rankings alone are not "
    "findings.",
    "Stretch ewma_recursive fits decay on train targets only and updates "
    "filter state on test rows using trailing RV features (not the forward "
    "label); it is distinct from frozen registry name ewma.",
)


@dataclass(frozen=True, slots=True)
class ImpliedVsRealizedSection:
    """Inputs for the HTML “Implied vs realized” section.

    Parameters
    ----------
    proxy_caveat : str
        VIX-as-IV-proxy research caveat.
    unit_conversion_note : str
        Locked ``vix_vol_daily`` formula note.
    vix_forecast_row : dict of str to object or None
        Horse-race row for ``vix_as_forecast`` (with ``comparison_note``),
        or ``None`` when the model was not screened.
    gap_feature_rows : tuple of dict
        Top IV−RV gap features from the ranking (may be empty).

    Methods
    -------
    has_vix_forecast()
        Return whether a VIX-as-forecast horse-race row is present.
    gap_feature_count()
        Return the number of gap-feature rows.
    """

    proxy_caveat: str
    unit_conversion_note: str
    vix_forecast_row: dict[str, object] | None
    gap_feature_rows: tuple[dict[str, object], ...]

    def has_vix_forecast(self) -> bool:
        """Return whether a VIX-as-forecast horse-race row is present."""
        return self.vix_forecast_row is not None

    def gap_feature_count(self) -> int:
        """Return the number of gap-feature rows."""
        return int(len(self.gap_feature_rows))


@dataclass(frozen=True, slots=True)
class ParametricVsHarSection:
    """Inputs for the HTML “Parametric vs HAR” section.

    Parameters
    ----------
    caveats : tuple of str
        Short discrete-OU / frozen-origin research caveats.
    ou_rv_row : dict of str to object or None
        Horse-race row for ``ou_rv`` (with ``comparison_note``), or ``None``
        when the model was not screened.
    filter_row : dict of str to object or None
        Optional stretch filter row (e.g. ``ewma_recursive``), or ``None``.

    Methods
    -------
    has_ou_rv()
        Return whether an ``ou_rv`` horse-race row is present.
    has_filter_model()
        Return whether a stretch filter horse-race row is present.
    """

    caveats: tuple[str, ...]
    ou_rv_row: dict[str, object] | None
    filter_row: dict[str, object] | None

    def has_ou_rv(self) -> bool:
        """Return whether an ``ou_rv`` horse-race row is present."""
        return self.ou_rv_row is not None

    def has_filter_model(self) -> bool:
        """Return whether a stretch filter horse-race row is present."""
        return self.filter_row is not None


@dataclass(frozen=True, slots=True)
class ReportExtras:
    """Non-tabular extras for a factor-screen memo.

    Parameters
    ----------
    importance_image_base64 : str or None
        Base64-encoded PNG bytes, if available.
    caveats : tuple of str
        Research caveats shown in the memo.
    implied_vs_realized : ImpliedVsRealizedSection or None
        Optional Implied vs realized section inputs.
    parametric_vs_har : ParametricVsHarSection or None
        Optional Parametric vs HAR section inputs.

    Methods
    -------
    has_image()
        Return whether an importance image is present.
    caveat_count()
        Return the number of caveat strings.
    """

    importance_image_base64: str | None
    caveats: tuple[str, ...]
    implied_vs_realized: ImpliedVsRealizedSection | None = None
    parametric_vs_har: ParametricVsHarSection | None = None

    def has_image(self) -> bool:
        """Return whether an importance image is present.

        Returns
        -------
        bool
            True when base64 image data exists.
        """
        return self.importance_image_base64 is not None

    def caveat_count(self) -> int:
        """Return the number of caveat strings.

        Returns
        -------
        int
            Length of ``caveats``.
        """
        return int(len(self.caveats))


def build_implied_vs_realized_section(
        model_rows: list[dict[str, object]],
        factor_rows: list[dict[str, object]],
) -> ImpliedVsRealizedSection:
    """Derive Implied vs realized inputs from horse-race and ranking rows.

    Parameters
    ----------
    model_rows : list of dict
        Horse-race rows after ``format_oos_gap_wording``.
    factor_rows : list of dict
        Factor-ranking rows (already importance-ordered).

    Returns
    -------
    ImpliedVsRealizedSection
        Section context; ``vix_forecast_row`` may be ``None``.
    """
    forecast_row = _find_model_row(model_rows, VIX_AS_FORECAST_MODEL)
    gap_rows = _top_iv_rv_gap_feature_rows(
        factor_rows,
        DEFAULT_TOP_IV_RV_GAP_FEATURES,
    )
    return ImpliedVsRealizedSection(
        proxy_caveat=VIX_PROXY_CAVEAT,
        unit_conversion_note=VIX_UNIT_CONVERSION_NOTE,
        vix_forecast_row=forecast_row,
        gap_feature_rows=tuple(gap_rows),
    )


def implied_section_template_payload(
        section: ImpliedVsRealizedSection | None,
) -> dict[str, object]:
    """Return Jinja keys for the Implied vs realized block."""
    if section is not None:
        return {
            "implied_proxy_caveat": section.proxy_caveat,
            "implied_unit_conversion_note": section.unit_conversion_note,
            "vix_forecast_row": section.vix_forecast_row,
            "iv_rv_gap_rows": list(section.gap_feature_rows),
        }
    return {
        "implied_proxy_caveat": VIX_PROXY_CAVEAT,
        "implied_unit_conversion_note": VIX_UNIT_CONVERSION_NOTE,
        "vix_forecast_row": None,
        "iv_rv_gap_rows": [],
    }


def parametric_section_template_payload(
        section: ParametricVsHarSection | None,
) -> dict[str, object]:
    """Return Jinja keys for the Parametric vs HAR block."""
    if section is not None:
        return {
            "parametric_caveats": list(section.caveats),
            "ou_rv_row": section.ou_rv_row,
            "parametric_filter_row": section.filter_row,
        }
    return {
        "parametric_caveats": list(PARAMETRIC_VS_HAR_CAVEATS),
        "ou_rv_row": None,
        "parametric_filter_row": None,
    }


def build_parametric_vs_har_section(
        model_rows: list[dict[str, object]],
) -> ParametricVsHarSection:
    """Derive Parametric vs HAR inputs from horse-race rows.

    Parameters
    ----------
    model_rows : list of dict
        Horse-race rows after ``format_oos_gap_wording``.

    Returns
    -------
    ParametricVsHarSection
        Section context; ``ou_rv_row`` / ``filter_row`` may be ``None``.
    """
    return ParametricVsHarSection(
        caveats=PARAMETRIC_VS_HAR_CAVEATS,
        ou_rv_row=_find_model_row(model_rows, OU_RV_MODEL),
        filter_row=_find_model_row(model_rows, STRETCH_FILTER_MODEL),
    )


def _top_iv_rv_gap_feature_rows(
        factor_rows: list[dict[str, object]],
        limit: int,
) -> list[dict[str, object]]:
    """Keep ranking rows whose feature name starts with ``vix_minus_rv_``."""
    matched = [
        row
        for row in factor_rows
        if str(row.get("feature", "")).startswith(IV_RV_GAP_FEATURE_PREFIX)
    ]
    return matched[:limit]


def _find_model_row(
        model_rows: list[dict[str, object]],
        model_name: str,
) -> dict[str, object] | None:
    """Return the first horse-race row matching ``model_name``, else None."""
    for row in model_rows:
        if str(row.get("model", "")) == model_name:
            return row
    return None
