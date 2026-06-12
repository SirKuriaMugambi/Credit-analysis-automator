"""ML-based credit scoring using logistic regression on synthetic training data."""

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
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
]


def generate_synthetic_training_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic company financial ratios for model training.

    ~70% healthy (label=0), ~30% distressed (label=1). Distributions are
    calibrated to realistic financial ratio ranges for each segment.
    """
    rng = np.random.default_rng(random_state)

    n_distressed = int(n_samples * 0.30)
    n_healthy = n_samples - n_distressed

    def clip(arr, low=None, high=None):
        return np.clip(arr, low, high)

    # ── Healthy segment ─────────────────────────────────────────────────────
    h_dso = clip(rng.normal(45, 15, n_healthy), 10, 90)
    h_dio = clip(rng.normal(50, 20, n_healthy), 10, 120)
    healthy = {
        "liquidity.current_ratio":     clip(rng.normal(2.0, 0.5,  n_healthy), 1.0),
        "liquidity.quick_ratio":       clip(rng.normal(1.5, 0.4,  n_healthy), 0.5),
        "liquidity.cash_ratio":        clip(rng.normal(0.7, 0.2,  n_healthy), 0.1),
        "leverage.debt_to_equity":     clip(rng.normal(0.5, 0.3,  n_healthy), 0.05, 1.8),
        "leverage.debt_to_assets":     clip(rng.normal(0.3, 0.1,  n_healthy), 0.05, 0.7),
        "leverage.equity_multiplier":  clip(rng.normal(1.5, 0.4,  n_healthy), 1.0),
        "leverage.interest_coverage":  clip(rng.normal(8.0, 3.0,  n_healthy), 2.0),
        "profitability.gross_margin":  clip(rng.normal(0.40, 0.10, n_healthy), 0.10, 0.80),
        "profitability.ebitda_margin": clip(rng.normal(0.20, 0.06, n_healthy), 0.05),
        "profitability.net_margin":    clip(rng.normal(0.12, 0.04, n_healthy), 0.02),
        "profitability.roa":           clip(rng.normal(0.10, 0.04, n_healthy), 0.01),
        "profitability.roe":           clip(rng.normal(0.15, 0.05, n_healthy), 0.01),
        "efficiency.asset_turnover":   clip(rng.normal(1.2,  0.4,  n_healthy), 0.3),
        "efficiency.dso":              h_dso,
        "efficiency.dio":              h_dio,
        "efficiency.cash_conversion_cycle": h_dso + h_dio,
        "cashflow.free_cash_flow":     rng.normal(50_000, 25_000, n_healthy),
        "cashflow.fcf_to_debt":        clip(rng.normal(0.20, 0.10, n_healthy), 0.01),
        "solvency.altman_z_score":     clip(rng.normal(3.5,  0.8,  n_healthy), 2.5),
    }
    healthy_df = pd.DataFrame(healthy)[FEATURE_NAMES]
    healthy_df["label"] = 0

    # ── Distressed segment ──────────────────────────────────────────────────
    d_dso = clip(rng.normal(90, 30, n_distressed), 30)
    d_dio = clip(rng.normal(120, 40, n_distressed), 30)
    distressed = {
        "liquidity.current_ratio":     clip(rng.normal(0.70, 0.20, n_distressed), 0.1,  1.2),
        "liquidity.quick_ratio":       clip(rng.normal(0.35, 0.15, n_distressed), 0.05, 0.7),
        "liquidity.cash_ratio":        clip(rng.normal(0.08, 0.05, n_distressed), 0.01, 0.5),
        "leverage.debt_to_equity":     clip(rng.normal(6.0,  2.0,  n_distressed), 2.5),
        "leverage.debt_to_assets":     clip(rng.normal(0.72, 0.12, n_distressed), 0.4,  0.95),
        "leverage.equity_multiplier":  clip(rng.normal(4.0,  1.5,  n_distressed), 2.0),
        "leverage.interest_coverage":  rng.normal(0.5,  0.6,  n_distressed),
        "profitability.gross_margin":  rng.normal(0.05, 0.10, n_distressed),
        "profitability.ebitda_margin": rng.normal(0.01, 0.06, n_distressed),
        "profitability.net_margin":    rng.normal(-0.06, 0.06, n_distressed),
        "profitability.roa":           rng.normal(-0.03, 0.04, n_distressed),
        "profitability.roe":           rng.normal(-0.15, 0.15, n_distressed),
        "efficiency.asset_turnover":   clip(rng.normal(0.35, 0.20, n_distressed), 0.05, 1.0),
        "efficiency.dso":              d_dso,
        "efficiency.dio":              d_dio,
        "efficiency.cash_conversion_cycle": d_dso + d_dio,
        "cashflow.free_cash_flow":     rng.normal(-10_000, 20_000, n_distressed),
        "cashflow.fcf_to_debt":        rng.normal(-0.05, 0.10, n_distressed),
        "solvency.altman_z_score":     clip(rng.normal(1.2,  0.4,  n_distressed), 0.1, 2.0),
    }
    distressed_df = pd.DataFrame(distressed)[FEATURE_NAMES]
    distressed_df["label"] = 1

    df = pd.concat([healthy_df, distressed_df], ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def train_credit_model(df: pd.DataFrame, model_path: str = "models/credit_model.pkl") -> dict:
    """
    Train a logistic regression credit model and save the fitted pipeline.

    Returns evaluation metrics and the feature name list used.
    """
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "feature_names": FEATURE_NAMES,
    }


def load_credit_model(model_path: str = "models/credit_model.pkl"):
    """Load a saved credit model pipeline from disk."""
    return joblib.load(model_path)


def predict_default_probability(fs, model_path: str = "models/credit_model.pkl") -> dict:
    """
    Predict probability of default for a FinancialStatements object.

    Returns probability_of_default (0-1), risk_band, and model_confidence.
    """
    from src.ratios import compute_all_ratios

    ratios = compute_all_ratios(fs)
    row = {k: ratios.get(k, float("nan")) for k in FEATURE_NAMES}
    X = pd.DataFrame([row])[FEATURE_NAMES]

    has_nan = any(math.isnan(v) for v in row.values() if isinstance(v, float))

    pipeline = load_credit_model(model_path)
    pd_score = float(pipeline.predict_proba(X)[0, 1])

    if pd_score < 0.2:
        risk_band = "Low"
    elif pd_score <= 0.5:
        risk_band = "Medium"
    else:
        risk_band = "High"

    return {
        "probability_of_default": pd_score,
        "risk_band": risk_band,
        "model_confidence": "Reduced" if has_nan else "High",
    }
