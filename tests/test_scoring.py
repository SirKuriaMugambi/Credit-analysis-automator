import math
import os

import pytest

from src.ratios import FinancialStatements


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_company():
    return FinancialStatements(
        current_assets=200_000, cash=80_000, inventory=40_000, receivables=80_000,
        total_assets=500_000, current_liabilities=100_000, total_liabilities=150_000,
        total_equity=350_000, retained_earnings=120_000, revenue=600_000, cogs=300_000,
        ebit=120_000, ebitda=150_000, interest_expense=15_000, net_income=90_000,
        operating_cash_flow=110_000, capex=20_000, market_cap=700_000,
    )


@pytest.fixture
def distressed_company():
    return FinancialStatements(
        current_assets=50_000, cash=5_000, inventory=30_000, receivables=15_000,
        total_assets=300_000, current_liabilities=80_000, total_liabilities=260_000,
        total_equity=40_000, retained_earnings=-20_000, revenue=150_000, cogs=140_000,
        ebit=5_000, ebitda=12_000, interest_expense=25_000, net_income=-10_000,
        operating_cash_flow=8_000, capex=15_000, market_cap=30_000,
    )


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    model_file = str(tmp / "credit_model.pkl")
    from src.scoring.ml_model import generate_synthetic_training_data, train_credit_model
    df = generate_synthetic_training_data(n_samples=600, random_state=42)
    train_credit_model(df, model_path=model_file)
    return model_file


# ─── PART A: rules_engine ─────────────────────────────────────────────────────

from src.scoring.rules_engine import classify_ratio, score_ratios, score_to_grade, RATIO_THRESHOLDS


class TestClassifyRatio:
    HIB = {"green": 1.5, "amber": 1.0, "higher_is_better": True}
    LOB = {"green": 1.0, "amber": 2.0, "higher_is_better": False}

    def test_classify_ratio_green_higher_is_better(self):
        assert classify_ratio(2.0, self.HIB) == "GREEN"

    def test_classify_ratio_amber_higher_is_better(self):
        assert classify_ratio(1.2, self.HIB) == "AMBER"

    def test_classify_ratio_red_higher_is_better(self):
        assert classify_ratio(0.5, self.HIB) == "RED"

    def test_classify_ratio_green_lower_is_better(self):
        assert classify_ratio(0.5, self.LOB) == "GREEN"

    def test_classify_ratio_amber_lower_is_better(self):
        assert classify_ratio(1.5, self.LOB) == "AMBER"

    def test_classify_ratio_red_lower_is_better(self):
        assert classify_ratio(3.0, self.LOB) == "RED"

    def test_classify_ratio_nan_is_red(self):
        assert classify_ratio(float("nan"), self.HIB) == "RED"

    def test_classify_ratio_none_is_red(self):
        assert classify_ratio(None, self.HIB) == "RED"

    def test_classify_ratio_at_green_boundary(self):
        assert classify_ratio(1.5, self.HIB) == "GREEN"

    def test_classify_ratio_at_amber_boundary(self):
        assert classify_ratio(1.0, self.HIB) == "AMBER"


class TestScoreToGradeBoundaries:
    def test_100(self):
        assert score_to_grade(100) == ("AAA", "Approve")

    def test_85(self):
        assert score_to_grade(85) == ("AAA", "Approve")

    def test_84_9(self):
        grade, rec = score_to_grade(84.9)
        assert grade == "AA"
        assert rec == "Approve"

    def test_75(self):
        assert score_to_grade(75) == ("AA", "Approve")

    def test_74_9(self):
        assert score_to_grade(74.9) == ("A", "Approve")

    def test_70(self):
        assert score_to_grade(70) == ("A", "Approve")

    def test_69_9(self):
        assert score_to_grade(69.9) == ("BBB", "Approve")

    def test_65(self):
        assert score_to_grade(65) == ("BBB", "Approve")

    def test_64_9(self):
        grade, rec = score_to_grade(64.9)
        assert grade == "BB"
        assert rec == "Approve with Conditions"

    def test_55(self):
        assert score_to_grade(55) == ("BB", "Approve with Conditions")

    def test_54_9(self):
        assert score_to_grade(54.9) == ("B", "Approve with Conditions")

    def test_45(self):
        assert score_to_grade(45) == ("B", "Approve with Conditions")

    def test_44_9(self):
        assert score_to_grade(44.9) == ("CCC", "Decline")

    def test_35(self):
        assert score_to_grade(35) == ("CCC", "Decline")

    def test_34_9(self):
        assert score_to_grade(34.9) == ("CC", "Decline")

    def test_20(self):
        assert score_to_grade(20) == ("CC", "Decline")

    def test_19_9(self):
        assert score_to_grade(19.9) == ("D", "Decline")

    def test_0(self):
        assert score_to_grade(0) == ("D", "Decline")


def test_score_ratios_returns_correct_keys(healthy_company):
    from src.ratios import compute_all_ratios
    ratios = compute_all_ratios(healthy_company)
    result = score_ratios(ratios)
    assert "total_score" in result
    assert "letter_grade" in result
    assert "recommendation" in result
    assert "ratio_breakdown" in result
    for key in RATIO_THRESHOLDS:
        assert key in result["ratio_breakdown"]
        breakdown = result["ratio_breakdown"][key]
        assert "value" in breakdown
        assert "status" in breakdown
        assert "points" in breakdown


def test_score_ratios_healthy_returns_high_grade(healthy_company):
    from src.ratios import compute_all_ratios
    ratios = compute_all_ratios(healthy_company)
    result = score_ratios(ratios)
    assert result["letter_grade"] in ("AAA", "AA", "A")
    assert result["recommendation"] == "Approve"
    assert result["total_score"] >= 70


def test_score_ratios_distressed_returns_low_grade(distressed_company):
    from src.ratios import compute_all_ratios
    ratios = compute_all_ratios(distressed_company)
    result = score_ratios(ratios)
    assert result["letter_grade"] in ("CCC", "CC", "D")
    assert result["recommendation"] == "Decline"
    assert result["total_score"] < 35


# ─── PART B: ml_model ─────────────────────────────────────────────────────────

from src.scoring.ml_model import (
    FEATURE_NAMES,
    generate_synthetic_training_data,
    load_credit_model,
    predict_default_probability,
    train_credit_model,
)


def test_generate_synthetic_training_data_shape_and_label_distribution():
    df = generate_synthetic_training_data(n_samples=1000, random_state=42)
    assert df.shape[0] == 1000
    assert df.shape[1] == len(FEATURE_NAMES) + 1
    assert "label" in df.columns
    dist = df["label"].value_counts(normalize=True)
    assert 0.25 <= dist[1] <= 0.35
    assert 0.65 <= dist[0] <= 0.75


def test_train_credit_model_runs_and_returns_metrics(tmp_path):
    df = generate_synthetic_training_data(n_samples=800, random_state=0)
    metrics = train_credit_model(df, model_path=str(tmp_path / "m.pkl"))
    assert metrics["roc_auc"] > 0.85
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert "feature_names" in metrics
    assert metrics["feature_names"] == FEATURE_NAMES


def test_train_credit_model_saves_file(tmp_path):
    df = generate_synthetic_training_data(n_samples=400, random_state=1)
    path = str(tmp_path / "credit_model.pkl")
    train_credit_model(df, model_path=path)
    assert os.path.exists(path)


def test_load_credit_model_works(trained_model_path):
    pipeline = load_credit_model(trained_model_path)
    assert hasattr(pipeline, "predict_proba")


def test_predict_default_probability_returns_valid_output(healthy_company, trained_model_path):
    result = predict_default_probability(healthy_company, model_path=trained_model_path)
    assert 0.0 <= result["probability_of_default"] <= 1.0
    assert result["risk_band"] in ("Low", "Medium", "High")
    assert result["model_confidence"] in ("High", "Reduced")


def test_predict_default_probability_on_distressed_returns_high_pd(distressed_company, trained_model_path):
    result = predict_default_probability(distressed_company, model_path=trained_model_path)
    assert result["probability_of_default"] > 0.5
    assert result["risk_band"] == "High"


# ─── PART C: combined ─────────────────────────────────────────────────────────

from src.scoring.combined import generate_combined_recommendation

EXPECTED_COMBINED_KEYS = {
    "ratios", "rules_based", "ml_based",
    "final_recommendation", "rationale", "warnings",
}


def test_combined_recommendation_returns_expected_keys(healthy_company, trained_model_path):
    result = generate_combined_recommendation(healthy_company, model_path=trained_model_path)
    assert EXPECTED_COMBINED_KEYS.issubset(result.keys())
    assert isinstance(result["warnings"], list)
    assert isinstance(result["rationale"], str)


def test_combined_recommendation_healthy_company_approves(healthy_company, trained_model_path):
    result = generate_combined_recommendation(healthy_company, model_path=trained_model_path)
    assert result["final_recommendation"] in ("Approve", "Approve with Conditions")


def test_combined_recommendation_distressed_company_declines(distressed_company, trained_model_path):
    result = generate_combined_recommendation(distressed_company, model_path=trained_model_path)
    assert result["final_recommendation"] == "Decline"


def test_combined_recommendation_disagreement_flagged_in_warnings(healthy_company, trained_model_path, monkeypatch):
    import src.scoring.combined as combined_module

    # Force ML to return high PD for a rules-based Approve company
    monkeypatch.setattr(
        combined_module,
        "predict_default_probability",
        lambda fs, model_path="models/credit_model.pkl": {
            "probability_of_default": 0.80,
            "risk_band": "High",
            "model_confidence": "High",
        },
    )

    result = combined_module.generate_combined_recommendation(
        healthy_company, model_path=trained_model_path
    )
    assert len(result["warnings"]) > 0
    assert any("Disagree" in w or "disagree" in w for w in result["warnings"])
