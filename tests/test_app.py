"""Smoke tests for app helper functions (no Streamlit rendering)."""

from __future__ import annotations

import math
import os
from dataclasses import fields

import pandas as pd
import pytest

from src.ratios import FinancialStatements
from src.app_helpers import (
    financial_statements_from_upload,
    format_ratio_value,
    get_sample_financials,
    get_template_dataframe,
    status_color,
)

_FS_FIELD_NAMES = {f.name for f in fields(FinancialStatements)}

_SAMPLE_EXCEL = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_financials.xlsx"
)


# ── get_sample_financials ─────────────────────────────────────────────────────

def test_get_sample_financials_returns_valid_object():
    fs = get_sample_financials()
    assert isinstance(fs, FinancialStatements)
    assert fs.current_assets is not None
    assert fs.total_assets is not None
    assert fs.revenue is not None
    assert fs.net_income is not None
    assert fs.total_equity is not None


# ── get_template_dataframe ────────────────────────────────────────────────────

def test_get_template_dataframe_has_expected_columns():
    df = get_template_dataframe()
    assert isinstance(df, pd.DataFrame)
    for name in _FS_FIELD_NAMES:
        assert name in df.columns, f"Missing column: {name}"


# ── financial_statements_from_upload ─────────────────────────────────────────

def test_financial_statements_from_upload_handles_missing_columns():
    sparse = pd.DataFrame([{"revenue": 500_000, "total_assets": 300_000}])
    fs = financial_statements_from_upload(sparse)
    assert isinstance(fs, FinancialStatements)
    assert fs.revenue == 500_000
    assert fs.total_assets == 300_000
    # Columns that were absent must be None (not raise)
    assert fs.cash is None
    assert fs.inventory is None


def test_financial_statements_from_upload_treats_nan_as_none():
    df = pd.DataFrame([{"revenue": float("nan"), "total_assets": 200_000}])
    fs = financial_statements_from_upload(df)
    assert fs.revenue is None
    assert fs.total_assets == 200_000


# ── format_ratio_value ────────────────────────────────────────────────────────

def test_format_ratio_value_formats_correctly():
    # Percentage ratios
    assert format_ratio_value("profitability.gross_margin", 0.45) == "45.0%"
    assert format_ratio_value("profitability.net_margin", 0.123) == "12.3%"
    assert format_ratio_value("cashflow.fcf_to_debt", 0.20) == "20.0%"

    # Days ratios
    assert format_ratio_value("efficiency.dso", 45.5) == "45.5 days"
    assert format_ratio_value("efficiency.dio", 60.0) == "60.0 days"
    assert format_ratio_value("efficiency.cash_conversion_cycle", 105.5) == "105.5 days"

    # Free cash flow — comma-formatted integer
    assert format_ratio_value("cashflow.free_cash_flow", 90_000) == "90,000"

    # Plain decimal ratios
    assert format_ratio_value("liquidity.current_ratio", 1.75) == "1.75"
    assert format_ratio_value("solvency.altman_z_score", 3.21) == "3.21"

    # NaN / None
    assert format_ratio_value("liquidity.current_ratio", float("nan")) == "N/A"
    assert format_ratio_value("profitability.roa", None) == "N/A"


# ── status_color ──────────────────────────────────────────────────────────────

def test_status_color_returns_valid_hex():
    for status in ("GREEN", "AMBER", "RED"):
        colour = status_color(status)
        assert colour.startswith("#"), f"Expected hex for {status}, got {colour}"
        assert len(colour) == 7, f"Hex should be 7 chars, got {colour}"

    # Unknown status returns a fallback hex, not an exception
    fallback = status_color("UNKNOWN")
    assert fallback.startswith("#")


# ── sample Excel file ─────────────────────────────────────────────────────────

def test_sample_excel_file_exists_and_has_expected_columns():
    assert os.path.exists(_SAMPLE_EXCEL), (
        f"data/sample_financials.xlsx not found at {_SAMPLE_EXCEL}"
    )
    df = pd.read_excel(_SAMPLE_EXCEL)
    assert len(df) >= 1, "Excel file should have at least one data row"
    for name in _FS_FIELD_NAMES:
        assert name in df.columns, f"Excel missing column: {name}"
