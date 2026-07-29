"""HTML report rendering for VIP experiments.

Exports
-------
render_factor_screen_report
    Render the factor-screen memo to an HTML string.
write_html_report
    Persist an HTML string to disk.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from vip.domain.errors import PersistenceError
from vip.reporting.experiment_summary import FactorScreenReportContext

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "factor_screen.html.j2"


def render_factor_screen_report(context: FactorScreenReportContext) -> str:
    """Render the factor-screen memo to an HTML string.

    Parameters
    ----------
    context : FactorScreenReportContext
        Render-ready template context.

    Returns
    -------
    str
        Complete HTML document.
    """
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    template = environment.get_template(TEMPLATE_NAME)
    return template.render(**context.as_template_dict())


def write_html_report(output_path: Path, html: str) -> Path:
    """Persist an HTML string to disk.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination ``.html`` path.
    html : str
        HTML document text.

    Returns
    -------
    pathlib.Path
        Path written to disk.

    Raises
    ------
    PersistenceError
        If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(html, encoding="utf-8")
    except OSError as exc:
        raise PersistenceError(
            f"Failed to write HTML report to {output_path}: {exc}"
        ) from exc
    return output_path