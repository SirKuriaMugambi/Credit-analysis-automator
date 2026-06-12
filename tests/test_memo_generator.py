import os

import pytest

from src.ratios import FinancialStatements
from src.memo_generator import CreditMemoGenerator


@pytest.fixture
def healthy_company():
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


def test_memo_generator_creates_pdf_file(healthy_company, tmp_path):
    output = str(tmp_path / "test_memo.pdf")
    gen = CreditMemoGenerator("Test Corp", healthy_company)
    result_path = gen.generate(output)
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 5 * 1024


def test_memo_generator_healthy_company_produces_valid_pdf(healthy_company, tmp_path):
    output = str(tmp_path / "healthy_memo.pdf")
    gen = CreditMemoGenerator("Healthy Corp Ltd", healthy_company)
    gen.generate(output)
    # Reportlab compresses PDFs; 7 KB is a safe floor for a 5-page memo
    assert os.path.getsize(output) > 7 * 1024


def test_memo_generator_distressed_company_produces_valid_pdf(distressed_company, tmp_path):
    output = str(tmp_path / "distressed_memo.pdf")
    gen = CreditMemoGenerator("Distressed Co", distressed_company)
    gen.generate(output)
    assert os.path.getsize(output) > 7 * 1024


def test_memo_generator_stores_analysis(healthy_company):
    gen = CreditMemoGenerator("Test Corp", healthy_company)
    for key in ("ratios", "rules_based", "ml_based", "final_recommendation"):
        assert key in gen.analysis


def test_memo_generator_default_analyst_is_caleb_mugambi(healthy_company):
    gen = CreditMemoGenerator("Test Corp", healthy_company)
    assert gen.analyst_name == "Caleb Mugambi"


def test_memo_generator_handles_nan_inputs(tmp_path):
    partial_fs = FinancialStatements(
        current_assets=100_000,
        current_liabilities=50_000,
        total_assets=200_000,
        total_liabilities=100_000,
        total_equity=100_000,
    )
    output = str(tmp_path / "nan_memo.pdf")
    gen = CreditMemoGenerator("Partial Data Inc", partial_fs)
    result_path = gen.generate(output)
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 5 * 1024
