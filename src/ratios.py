from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatements:
    # Balance sheet
    current_assets: Optional[float] = None
    cash: Optional[float] = None
    inventory: Optional[float] = None
    receivables: Optional[float] = None
    total_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    retained_earnings: Optional[float] = None
    working_capital: Optional[float] = None  # computed if not provided

    # Income statement
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    net_income: Optional[float] = None
    tax_expense: Optional[float] = None

    # Cash flow
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    # Market data
    market_cap: Optional[float] = None

    def __post_init__(self):
        if self.working_capital is None and self.current_assets is not None and self.current_liabilities is not None:
            self.working_capital = self.current_assets - self.current_liabilities


def _safe(numerator, denominator) -> float:
    """Return numerator/denominator, or nan with a warning if inputs are invalid."""
    if numerator is None or denominator is None:
        logger.warning("Missing required input for ratio calculation.")
        return float("nan")
    if denominator == 0:
        logger.warning("Division by zero in ratio calculation.")
        return float("nan")
    return numerator / denominator


# ─── LIQUIDITY ────────────────────────────────────────────────────────────────

def current_ratio(fs: FinancialStatements) -> float:
    return _safe(fs.current_assets, fs.current_liabilities)


def quick_ratio(fs: FinancialStatements) -> float:
    if fs.current_assets is None or fs.inventory is None:
        logger.warning("Missing required input for quick_ratio.")
        return float("nan")
    return _safe(fs.current_assets - fs.inventory, fs.current_liabilities)


def cash_ratio(fs: FinancialStatements) -> float:
    return _safe(fs.cash, fs.current_liabilities)


# ─── LEVERAGE ─────────────────────────────────────────────────────────────────

def debt_to_equity(fs: FinancialStatements) -> float:
    return _safe(fs.total_liabilities, fs.total_equity)


def debt_to_assets(fs: FinancialStatements) -> float:
    return _safe(fs.total_liabilities, fs.total_assets)


def equity_multiplier(fs: FinancialStatements) -> float:
    return _safe(fs.total_assets, fs.total_equity)


def interest_coverage(fs: FinancialStatements) -> float:
    return _safe(fs.ebit, fs.interest_expense)


# ─── PROFITABILITY ────────────────────────────────────────────────────────────

def gross_margin(fs: FinancialStatements) -> float:
    if fs.revenue is None or fs.cogs is None:
        logger.warning("Missing required input for gross_margin.")
        return float("nan")
    return _safe(fs.revenue - fs.cogs, fs.revenue)


def ebitda_margin(fs: FinancialStatements) -> float:
    return _safe(fs.ebitda, fs.revenue)


def net_margin(fs: FinancialStatements) -> float:
    return _safe(fs.net_income, fs.revenue)


def roa(fs: FinancialStatements) -> float:
    return _safe(fs.net_income, fs.total_assets)


def roe(fs: FinancialStatements) -> float:
    return _safe(fs.net_income, fs.total_equity)


# ─── EFFICIENCY ───────────────────────────────────────────────────────────────

def asset_turnover(fs: FinancialStatements) -> float:
    return _safe(fs.revenue, fs.total_assets)


def dso(fs: FinancialStatements) -> float:
    """Days Sales Outstanding = (receivables / revenue) * 365"""
    if fs.receivables is None or fs.revenue is None:
        logger.warning("Missing required input for dso.")
        return float("nan")
    return _safe(fs.receivables * 365, fs.revenue)


def dio(fs: FinancialStatements) -> float:
    """Days Inventory Outstanding = (inventory / cogs) * 365"""
    if fs.inventory is None or fs.cogs is None:
        logger.warning("Missing required input for dio.")
        return float("nan")
    return _safe(fs.inventory * 365, fs.cogs)


def cash_conversion_cycle(fs: FinancialStatements) -> float:
    """DSO + DIO - DPO (DPO assumed 0 if not provided)"""
    d = dso(fs)
    di = dio(fs)
    if math.isnan(d) or math.isnan(di):
        return float("nan")
    return d + di


# ─── CASH FLOW ────────────────────────────────────────────────────────────────

def free_cash_flow(fs: FinancialStatements) -> float:
    """operating_cash_flow - capex"""
    if fs.operating_cash_flow is None or fs.capex is None:
        logger.warning("Missing required input for free_cash_flow.")
        return float("nan")
    return float(fs.operating_cash_flow - fs.capex)


def fcf_to_debt(fs: FinancialStatements) -> float:
    fcf = free_cash_flow(fs)
    if math.isnan(fcf):
        return float("nan")
    return _safe(fcf, fs.total_liabilities)


# ─── SOLVENCY / DISTRESS ──────────────────────────────────────────────────────

def altman_z_score(fs: FinancialStatements) -> float:
    """
    Classic Altman Z-Score for public companies.

    Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

    Components:
      A = working_capital / total_assets       (short-term liquidity)
      B = retained_earnings / total_assets     (cumulative profitability)
      C = ebit / total_assets                  (operating efficiency)
      D = market_cap / total_liabilities       (market-based solvency)
      E = revenue / total_assets               (asset utilization)

    Interpretation thresholds:
      Z > 2.99  → Safe zone
      1.81–2.99 → Grey zone (uncertainty)
      Z < 1.81  → Distress zone
    """
    required = [
        fs.working_capital, fs.total_assets, fs.retained_earnings,
        fs.ebit, fs.market_cap, fs.total_liabilities, fs.revenue,
    ]
    if any(v is None for v in required):
        logger.warning("Missing required input for altman_z_score.")
        return float("nan")
    if fs.total_assets == 0 or fs.total_liabilities == 0:
        logger.warning("Division by zero in altman_z_score.")
        return float("nan")

    A = fs.working_capital / fs.total_assets
    B = fs.retained_earnings / fs.total_assets
    C = fs.ebit / fs.total_assets
    D = fs.market_cap / fs.total_liabilities
    E = fs.revenue / fs.total_assets

    return 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E


# ─── MASTER FUNCTION ──────────────────────────────────────────────────────────

def compute_all_ratios(fs: FinancialStatements) -> dict[str, float]:
    """Compute every ratio and return a flat dict with dot-notation keys."""
    return {
        "liquidity.current_ratio": current_ratio(fs),
        "liquidity.quick_ratio": quick_ratio(fs),
        "liquidity.cash_ratio": cash_ratio(fs),
        "leverage.debt_to_equity": debt_to_equity(fs),
        "leverage.debt_to_assets": debt_to_assets(fs),
        "leverage.equity_multiplier": equity_multiplier(fs),
        "leverage.interest_coverage": interest_coverage(fs),
        "profitability.gross_margin": gross_margin(fs),
        "profitability.ebitda_margin": ebitda_margin(fs),
        "profitability.net_margin": net_margin(fs),
        "profitability.roa": roa(fs),
        "profitability.roe": roe(fs),
        "efficiency.asset_turnover": asset_turnover(fs),
        "efficiency.dso": dso(fs),
        "efficiency.dio": dio(fs),
        "efficiency.cash_conversion_cycle": cash_conversion_cycle(fs),
        "cashflow.free_cash_flow": free_cash_flow(fs),
        "cashflow.fcf_to_debt": fcf_to_debt(fs),
        "solvency.altman_z_score": altman_z_score(fs),
    }
