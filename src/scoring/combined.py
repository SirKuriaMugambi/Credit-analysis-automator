"""Combined credit recommendation merging rules-based scorecard and ML signals."""

import math

from src.ratios import compute_all_ratios
from src.scoring.rules_engine import score_ratios
from src.scoring.ml_model import predict_default_probability


def generate_combined_recommendation(fs, model_path: str = "models/credit_model.pkl") -> dict:
    """
    Produce a unified credit recommendation by combining rules-based and ML scoring.

    Decision logic:
      - Both rules say "Approve" AND ML PD < 0.30  ->  "Approve"
      - Rules say "Decline"  OR  ML PD > 0.60      ->  "Decline"
      - Otherwise                                   ->  "Approve with Conditions"

    Returns ratios, rules_based result, ml_based result, final_recommendation,
    rationale, and a list of warnings (disagreements, missing data).
    """
    ratios = compute_all_ratios(fs)
    rules_result = score_ratios(ratios)
    ml_result = predict_default_probability(fs, model_path=model_path)

    rules_rec = rules_result["recommendation"]
    pd_score = ml_result["probability_of_default"]
    ml_risk = ml_result["risk_band"]

    if rules_rec == "Approve" and pd_score < 0.30:
        final_rec = "Approve"
    elif rules_rec == "Decline" or pd_score > 0.60:
        final_rec = "Decline"
    else:
        final_rec = "Approve with Conditions"

    grade = rules_result["letter_grade"]
    rationale = (
        f"Rules-based scorecard assigns grade {grade} "
        f"(score {rules_result['total_score']:.1f}/100). "
        f"ML model estimates a {pd_score:.1%} probability of default ({ml_risk} risk)."
    )

    warnings = []

    if rules_rec == "Approve" and ml_risk == "High":
        warnings.append(
            "Disagreement: rules-based engine recommends Approve but ML signals High default risk."
        )
    elif rules_rec == "Decline" and ml_risk == "Low":
        warnings.append(
            "Disagreement: rules-based engine recommends Decline but ML signals Low default risk."
        )

    nan_keys = [k for k, v in ratios.items() if isinstance(v, float) and math.isnan(v)]
    if nan_keys:
        warnings.append(
            f"Missing data for {len(nan_keys)} ratio(s): {', '.join(nan_keys)}."
        )

    if ml_result["model_confidence"] == "Reduced":
        warnings.append("ML model confidence is Reduced due to missing input features.")

    return {
        "ratios": ratios,
        "rules_based": rules_result,
        "ml_based": ml_result,
        "final_recommendation": final_rec,
        "rationale": rationale,
        "warnings": warnings,
    }
