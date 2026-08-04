"""Tests for the VIP CLI."""

from typer.testing import CliRunner

from vip.cli.main import app

runner = CliRunner()


def test_help_exits_zero() -> None:
    """Root help should succeed."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Volatility Intelligence Platform" in result.stdout


def test_info_prints_defaults() -> None:
    """Info command should surface locked default research settings."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "vip 0.1.0" in result.stdout
    assert "symbol: SPY" in result.stdout
    assert "horizon_days: 5" in result.stdout
    assert "primary_metric: qlike" in result.stdout


def test_ingest_help_exits_zero() -> None:
    """Ingest help should be available."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.stdout


def test_features_help_exits_zero() -> None:
    """Features help should be available."""
    result = runner.invoke(app, ["features", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.stdout
    assert "--horizon" in result.stdout


def test_evaluate_help_exits_zero() -> None:
    """Evaluate help should be available."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.stdout
    assert "--n-splits" in result.stdout
    assert "--embargo" in result.stdout


def test_screen_horizons_help_exits_zero() -> None:
    """Multi-horizon screen help should be available."""
    result = runner.invoke(app, ["screen-horizons", "--help"])
    assert result.exit_code == 0
    assert "--symbol" in result.stdout
    assert "--horizons" in result.stdout
    assert "--with-vix" in result.stdout
    assert "--skip-features" in result.stdout