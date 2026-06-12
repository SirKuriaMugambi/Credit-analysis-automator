"""
Rules-based credit scoring engine using a transparent scorecard methodology.

Philosophy: Each financial ratio is assessed against industry-standard thresholds
and classified as GREEN (strong), AMBER (borderline), or RED (weak). The total
score is a weighted average across 9 key ratios (equal weights) that maps to a
letter grade and lending recommendation.

Output interpretation:
  AAA/AA/A (score 70+)  : Approve — strong creditworthiness
  BBB       (score 65-69): Approve — acceptable with monitoring
  BB/B      (score 45-64): Approve with Conditions — covenants required
  CCC/CC/D  (score < 45) : Decline — credit risk too high

NaN values are treated as RED (missing data is a negative signal).
"""

import math
from enum import Enum


class RatioStatus(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


RATIO_THRESHOLDS = {
    "liquidity.current_ratio":      {"green": 1.5,   "amber": 1.0,   "higher_is_better": True},
    "liquidity.quick_ratio":        {"green": 1.0,   "amber": 0.7,   "higher_is_better": True},
    "leverage.debt_to_equity":      {"green": 1.0,   "amber": 2.0,   "higher_is_better": False},
    "leverage.interest_coverage":   {"green": 3.0,   "amber": 1.5,   "higher_is_better": True},
    "profitability.ebitda_margin":  {"green": 0.15,  "amber": 0.05,  "higher_is_better": True},
    "profitability.net_margin":     {"green": 0.10,  "amber": 0.03,  "higher_is_better": True},
    "profitability.roa":            {"green": 0.08,  "amber": 0.02,  "higher_is_better": True},
    "efficiency.asset_turnover":    {"green": 1.0,   "amber": 0.5,   "higher_is_better": True},
    "solvency.altman_z_score":      {"green": 2.99,  "amber": 1.81,  "higher_is_better": True},
}

_POINTS_PER_RATIO = 100.0 / len(RATIO_THRESHOLDS)


def classify_ratio(value: float, thresholds: dict) -> str:
    """Return GREEN, AMBER, or RED for a single ratio value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return RatioStatus.RED.value

    higher_is_better = thresholds["higher_is_better"]
    green = thresholds["green"]
    amber = thresholds["amber"]

    if higher_is_better:
        if value >= green:
            return RatioStatus.GREEN.value
        if value >= amber:
            return RatioStatus.AMBER.value
        return RatioStatus.RED.value
    else:
        if value <= green:
            return RatioStatus.GREEN.value
        if value <= amber:
            return RatioStatus.AMBER.value
        return RatioStatus.RED.value


def score_to_grade(total_score: float) -> tuple:
    """Map a 0-100 score to (letter_grade, recommendation)."""
    if total_score >= 85:
        return "AAA", "Approve"
    if total_score >= 75:
        return "AA", "Approve"
    if total_score >= 70:
        return "A", "Approve"
    if total_score >= 65:
        return "BBB", "Approve"
    if total_score >= 55:
        return "BB", "Approve with Conditions"
    if total_score >= 45:
        return "B", "Approve with Conditions"
    if total_score >= 35:
        return "CCC", "Decline"
    if total_score >= 20:
        return "CC", "Decline"
    return "D", "Decline"


def score_ratios(ratios: dict) -> dict:
    """
    Score a ratio dict against the scorecard and return a structured result.

    Returns:
        total_score      : float 0-100
        letter_grade     : one of AAA AA A BBB BB B CCC CC D
        recommendation   : "Approve" | "Approve with Conditions" | "Decline"
        ratio_breakdown  : {ratio_key: {value, status, points}}
    """
    breakdown = {}
    total_score = 0.0

    for ratio_key, thresholds in RATIO_THRESHOLDS.items():
        value = ratios.get(ratio_key, float("nan"))
        status = classify_ratio(value, thresholds)

        if status == RatioStatus.GREEN.value:
            points = _POINTS_PER_RATIO
        elif status == RatioStatus.AMBER.value:
            points = _POINTS_PER_RATIO / 2.0
        else:
            points = 0.0

        total_score += points
        breakdown[ratio_key] = {
            "value": value,
            "status": status,
            "points": round(points, 4),
        }

    letter_grade, recommendation = score_to_grade(total_score)

    return {
        "total_score": round(total_score, 4),
        "letter_grade": letter_grade,
        "recommendation": recommendation,
        "ratio_breakdown": breakdown,
    }
