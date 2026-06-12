"""Pure-Python helper functions for the Streamlit app — importable without Streamlit."""

from __future__ import annotations

import math
from dataclasses import fields

import pandas as pd

from src.ratios import FinancialStatements

# ── Ratio classification sets ──────────────────────────────────────────────────

PERCENT_RATIOS: frozenset[str] = frozenset({
    "profitability.gross_margin",
    "profitability.ebitda_margin",
    "profitability.net_margin",
    "profitability.roa",
    "profitability.roe",
    "cashflow.fcf_to_debt",
})

DAYS_RATIOS: frozenset[str] = frozenset({
    "efficiency.dso",
    "efficiency.dio",
    "efficiency.cash_conversion_cycle",
})

RATIO_LABELS: dict[str, str] = {
    "liquidity.current_ratio":          "Current Ratio",
    "liquidity.quick_ratio":            "Quick Ratio",
    "liquidity.cash_ratio":             "Cash Ratio",
    "leverage.debt_to_equity":          "Debt to Equity",
    "leverage.debt_to_assets":          "Debt to Assets",
    "leverage.equity_multiplier":       "Equity Multiplier",
    "leverage.interest_coverage":       "Interest Coverage",
    "profitability.gross_margin":       "Gross Margin",
    "profitability.ebitda_margin":      "EBITDA Margin",
    "profitability.net_margin":         "Net Margin",
    "profitability.roa":                "Return on Assets",
    "profitability.roe":                "Return on Equity",
    "efficiency.asset_turnover":        "Asset Turnover",
    "efficiency.dso":                   "Days Sales Outstanding",
    "efficiency.dio":                   "Days Inventory Outstanding",
    "efficiency.cash_conversion_cycle": "Cash Conversion Cycle",
    "cashflow.free_cash_flow":          "Free Cash Flow",
    "cashflow.fcf_to_debt":             "FCF to Debt",
    "solvency.altman_z_score":          "Altman Z-Score",
}

CATEGORIES: dict[str, list[str]] = {
    "Liquidity": [
        "liquidity.current_ratio",
        "liquidity.quick_ratio",
        "liquidity.cash_ratio",
    ],
    "Leverage": [
        "leverage.debt_to_equity",
        "leverage.debt_to_assets",
        "leverage.equity_multiplier",
        "leverage.interest_coverage",
    ],
    "Profitability": [
        "profitability.gross_margin",
        "profitability.ebitda_margin",
        "profitability.net_margin",
        "profitability.roa",
        "profitability.roe",
    ],
    "Efficiency": [
        "efficiency.asset_turnover",
        "efficiency.dso",
        "efficiency.dio",
        "efficiency.cash_conversion_cycle",
    ],
    "Cash Flow": [
        "cashflow.free_cash_flow",
        "cashflow.fcf_to_debt",
    ],
    "Solvency": [
        "solvency.altman_z_score",
    ],
}

# ── Helper functions ───────────────────────────────────────────────────────────

def get_sample_financials() -> FinancialStatements:
    """Return the Green Solutions Manufacturing Ltd healthy-company financials."""
    return FinancialStatements(
        current_assets=200_000,
        cash=80_000,
        inventory=40_000,
        receivables=80_000,
        total_assets=500_000,
        current_liabilities=100_000,
        total_liabilities=150_000,
        total_equity=350_000,
        retained_earnings=120_000,
        revenue=600_000,
        cogs=300_000,
        ebit=120_000,
        ebitda=150_000,
        interest_expense=15_000,
        net_income=90_000,
        tax_expense=30_000,
        operating_cash_flow=110_000,
        capex=20_000,
        market_cap=700_000,
    )


def get_template_dataframe() -> pd.DataFrame:
    """Return a one-row template DataFrame with all FinancialStatements fields populated."""
    fs = get_sample_financials()
    row = {f.name: getattr(fs, f.name) for f in fields(FinancialStatements)}
    return pd.DataFrame([row])


def financial_statements_from_upload(df: pd.DataFrame) -> FinancialStatements:
    """
    Build a FinancialStatements from the first row of an uploaded DataFrame.
    Missing columns become None; present but NaN values also become None.
    """
    row = df.iloc[0]
    kwargs: dict = {}
    for f in fields(FinancialStatements):
        if f.name in row.index:
            val = row[f.name]
            kwargs[f.name] = None if (val is None or (isinstance(val, float) and math.isnan(val))) else float(val)
        else:
            kwargs[f.name] = None
    return FinancialStatements(**kwargs)


def format_ratio_value(ratio_key: str, value: float) -> str:
    """Format a ratio value for human-readable display."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if ratio_key in PERCENT_RATIOS:
        return f"{value * 100:.1f}%"
    if ratio_key in DAYS_RATIOS:
        return f"{value:.1f} days"
    if ratio_key == "cashflow.free_cash_flow":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def status_color(status: str) -> str:
    """Map GREEN / AMBER / RED to a hex colour string."""
    return {
        "GREEN": "#90EE90",
        "AMBER": "#FFD580",
        "RED":   "#FF7F7F",
    }.get(status, "#CCCCCC")
