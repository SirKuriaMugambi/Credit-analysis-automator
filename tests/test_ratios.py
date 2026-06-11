import math
import pytest

from src.ratios import (
    FinancialStatements,
    altman_z_score,
    asset_turnover,
    cash_conversion_cycle,
    cash_ratio,
    compute_all_ratios,
    current_ratio,
    debt_to_assets,
    debt_to_equity,
    dio,
    dso,
    ebitda_margin,
    equity_multiplier,
    fcf_to_debt,
    free_cash_flow,
    gross_margin,
    interest_coverage,
    net_margin,
    quick_ratio,
    roa,
    roe,
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_company():
    """Well-performing company: current ratio ~2, D/E ~0.5, Altman Z > 3."""
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
        operating_cash_flow=110_000,
        capex=20_000,
        market_cap=700_000,
    )


@pytest.fixture
def distressed_company():
    """Distressed company: current ratio < 1, high D/E, Altman Z < 1.81."""
    return FinancialStatements(
        current_assets=50_000,
        cash=5_000,
        inventory=30_000,
        receivables=15_000,
        total_assets=300_000,
        current_liabilities=80_000,
        total_liabilities=260_000,
        total_equity=40_000,
        retained_earnings=-20_000,
        revenue=150_000,
        cogs=140_000,
        ebit=5_000,
        ebitda=12_000,
        interest_expense=25_000,
        net_income=-10_000,
        operating_cash_flow=8_000,
        capex=15_000,
        market_cap=30_000,
    )


@pytest.fixture
def zero_division_company():
    """Edge case: key denominators are zero."""
    return FinancialStatements(
        current_assets=100_000,
        cash=10_000,
        inventory=20_000,
        receivables=30_000,
        total_assets=0,
        current_liabilities=0,
        total_liabilities=0,
        total_equity=0,
        retained_earnings=5_000,
        revenue=0,
        cogs=0,
        ebit=10_000,
        ebitda=15_000,
        interest_expense=0,
        net_income=5_000,
        operating_cash_flow=8_000,
        capex=3_000,
        market_cap=50_000,
    )


# ─── LIQUIDITY ────────────────────────────────────────────────────────────────

def test_current_ratio_healthy(healthy_company):
    assert current_ratio(healthy_company) == pytest.approx(2.0)


def test_quick_ratio_healthy(healthy_company):
    # (200k - 40k) / 100k = 1.6
    assert quick_ratio(healthy_company) == pytest.approx(1.6)


def test_cash_ratio_healthy(healthy_company):
    # 80k / 100k = 0.8
    assert cash_ratio(healthy_company) == pytest.approx(0.8)


def test_current_ratio_distressed(distressed_company):
    # 50k / 80k = 0.625  < 1
    assert current_ratio(distressed_company) < 1.0


def test_current_ratio_zero_denom(zero_division_company):
    assert math.isnan(current_ratio(zero_division_company))


def test_quick_ratio_zero_denom(zero_division_company):
    assert math.isnan(quick_ratio(zero_division_company))


def test_cash_ratio_zero_denom(zero_division_company):
    assert math.isnan(cash_ratio(zero_division_company))


# ─── LEVERAGE ─────────────────────────────────────────────────────────────────

def test_debt_to_equity_healthy(healthy_company):
    # 150k / 350k ≈ 0.4286
    assert debt_to_equity(healthy_company) == pytest.approx(150_000 / 350_000)


def test_debt_to_equity_distressed(distressed_company):
    # 260k / 40k = 6.5 — much higher
    assert debt_to_equity(distressed_company) > 5.0


def test_debt_to_assets_healthy(healthy_company):
    assert debt_to_assets(healthy_company) == pytest.approx(150_000 / 500_000)


def test_equity_multiplier_healthy(healthy_company):
    assert equity_multiplier(healthy_company) == pytest.approx(500_000 / 350_000)


def test_interest_coverage_healthy(healthy_company):
    # 120k / 15k = 8.0
    assert interest_coverage(healthy_company) == pytest.approx(8.0)


def test_interest_coverage_zero_denom(zero_division_company):
    assert math.isnan(interest_coverage(zero_division_company))


def test_debt_to_equity_zero_denom(zero_division_company):
    assert math.isnan(debt_to_equity(zero_division_company))


# ─── PROFITABILITY ────────────────────────────────────────────────────────────

def test_gross_margin_healthy(healthy_company):
    # (600k - 300k) / 600k = 0.5
    assert gross_margin(healthy_company) == pytest.approx(0.5)


def test_ebitda_margin_healthy(healthy_company):
    # 150k / 600k = 0.25
    assert ebitda_margin(healthy_company) == pytest.approx(0.25)


def test_net_margin_healthy(healthy_company):
    # 90k / 600k = 0.15
    assert net_margin(healthy_company) == pytest.approx(0.15)


def test_roa_healthy(healthy_company):
    # 90k / 500k = 0.18
    assert roa(healthy_company) == pytest.approx(0.18)


def test_roe_healthy(healthy_company):
    # 90k / 350k ≈ 0.2571
    assert roe(healthy_company) == pytest.approx(90_000 / 350_000)


def test_net_margin_distressed(distressed_company):
    assert net_margin(distressed_company) < 0


def test_gross_margin_zero_denom(zero_division_company):
    assert math.isnan(gross_margin(zero_division_company))


def test_ebitda_margin_zero_denom(zero_division_company):
    assert math.isnan(ebitda_margin(zero_division_company))


# ─── EFFICIENCY ───────────────────────────────────────────────────────────────

def test_asset_turnover_healthy(healthy_company):
    # 600k / 500k = 1.2
    assert asset_turnover(healthy_company) == pytest.approx(1.2)


def test_dso_healthy(healthy_company):
    # (80k / 600k) * 365 ≈ 48.67
    assert dso(healthy_company) == pytest.approx((80_000 / 600_000) * 365)


def test_dio_healthy(healthy_company):
    # (40k / 300k) * 365 ≈ 48.67
    assert dio(healthy_company) == pytest.approx((40_000 / 300_000) * 365)


def test_cash_conversion_cycle_healthy(healthy_company):
    expected = dso(healthy_company) + dio(healthy_company)
    assert cash_conversion_cycle(healthy_company) == pytest.approx(expected)


def test_dso_zero_denom(zero_division_company):
    assert math.isnan(dso(zero_division_company))


def test_dio_zero_denom(zero_division_company):
    assert math.isnan(dio(zero_division_company))


def test_ccc_zero_denom(zero_division_company):
    assert math.isnan(cash_conversion_cycle(zero_division_company))


# ─── CASH FLOW ────────────────────────────────────────────────────────────────

def test_free_cash_flow_healthy(healthy_company):
    # 110k - 20k = 90k
    assert free_cash_flow(healthy_company) == pytest.approx(90_000)


def test_fcf_to_debt_healthy(healthy_company):
    # 90k / 150k = 0.6
    assert fcf_to_debt(healthy_company) == pytest.approx(0.6)


def test_free_cash_flow_distressed(distressed_company):
    # 8k - 15k = -7k (negative FCF)
    assert free_cash_flow(distressed_company) < 0


def test_fcf_to_debt_zero_denom(zero_division_company):
    assert math.isnan(fcf_to_debt(zero_division_company))


# ─── ALTMAN Z-SCORE ───────────────────────────────────────────────────────────

def test_altman_z_healthy(healthy_company):
    z = altman_z_score(healthy_company)
    assert not math.isnan(z)
    assert z > 2.99  # Safe zone


def test_altman_z_distressed(distressed_company):
    z = altman_z_score(distressed_company)
    assert not math.isnan(z)
    assert z < 1.81  # Distress zone


def test_altman_z_known_example():
    """Worked example with hand-calculated expected Z-score."""
    fs = FinancialStatements(
        working_capital=100,
        total_assets=500,
        retained_earnings=200,
        ebit=60,
        market_cap=300,
        total_liabilities=200,
        revenue=400,
    )
    # A = 100/500 = 0.2
    # B = 200/500 = 0.4
    # C =  60/500 = 0.12
    # D = 300/200 = 1.5
    # E = 400/500 = 0.8
    # Z = 1.2*0.2 + 1.4*0.4 + 3.3*0.12 + 0.6*1.5 + 1.0*0.8
    #   = 0.24 + 0.56 + 0.396 + 0.9 + 0.8 = 2.896
    assert altman_z_score(fs) == pytest.approx(2.896)


def test_altman_z_zero_denom(zero_division_company):
    assert math.isnan(altman_z_score(zero_division_company))


def test_altman_z_missing_input():
    fs = FinancialStatements(total_assets=500)  # all others None
    assert math.isnan(altman_z_score(fs))


# ─── COMPUTE ALL RATIOS ───────────────────────────────────────────────────────

EXPECTED_KEYS = {
    "liquidity.current_ratio",
    "liquidity.quick_ratio",
    "liquidity.cash_ratio",
    "leverage.debt_to_equity",
    "leverage.debt_to_assets",
    "leverage.equity_multiplier",
    "leverage.interest_coverage",
    "profitability.gross_margin",
    "profitability.ebitda_margin",
    "profitability.net_margin",
    "profitability.roa",
    "profitability.roe",
    "efficiency.asset_turnover",
    "efficiency.dso",
    "efficiency.dio",
    "efficiency.cash_conversion_cycle",
    "cashflow.free_cash_flow",
    "cashflow.fcf_to_debt",
    "solvency.altman_z_score",
}


def test_compute_all_ratios_keys(healthy_company):
    result = compute_all_ratios(healthy_company)
    assert EXPECTED_KEYS.issubset(result.keys())


def test_compute_all_ratios_values_are_floats(healthy_company):
    result = compute_all_ratios(healthy_company)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_compute_all_ratios_distressed_keys(distressed_company):
    result = compute_all_ratios(distressed_company)
    assert EXPECTED_KEYS.issubset(result.keys())
