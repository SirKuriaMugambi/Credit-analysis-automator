"""
PDF Credit Memo Generator for the Credit Analysis Automator.

Generates a multi-section A4 PDF credit memo including:
  - Cover page with document metadata
  - Executive summary with key scores and final recommendation
  - Financial analysis with ratio tables across six categories
  - Risk assessment with Altman Z-Score and key risk factors
  - Recommendation with optional covenant conditions and analyst sign-off
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,  # noqa: F401 – imported per spec; available for future chart embedding
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.ratios import FinancialStatements
from src.scoring.combined import generate_combined_recommendation
from src.scoring.rules_engine import RATIO_THRESHOLDS

# ── Colour palette ─────────────────────────────────────────────────────────────
_NAVY   = colors.HexColor("#1a365d")
_WHITE  = colors.white
_LGREY  = colors.HexColor("#f5f5f5")
_DKGREY = colors.HexColor("#333333")
_GREEN  = colors.HexColor("#90EE90")
_AMBER  = colors.HexColor("#FFD580")
_RED    = colors.HexColor("#FF7F7F")

_GRADE_COLORS = {
    "AAA": _GREEN, "AA": _GREEN, "A": _GREEN, "BBB": _GREEN,
    "BB": _AMBER, "B": _AMBER,
    "CCC": _RED, "CC": _RED, "D": _RED,
}
_RISK_BAND_COLORS = {"Low": _GREEN, "Medium": _AMBER, "High": _RED}

# ── Ratio groupings (display order) ───────────────────────────────────────────
_CATEGORIES = {
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

_RATIO_LABELS = {
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

_PERCENT_RATIOS = {
    "profitability.gross_margin",
    "profitability.ebitda_margin",
    "profitability.net_margin",
    "profitability.roa",
    "profitability.roe",
    "cashflow.fcf_to_debt",
}

_DAYS_RATIOS = {
    "efficiency.dso",
    "efficiency.dio",
    "efficiency.cash_conversion_cycle",
}


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_value(key: str, value) -> str:
    if _is_nan(value):
        return "N/A"
    if key in _PERCENT_RATIOS:
        return f"{value * 100:.1f}%"
    if key in _DAYS_RATIOS:
        return f"{value:.1f} days"
    if key == "cashflow.free_cash_flow":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _fmt_threshold(key: str) -> str:
    if key not in RATIO_THRESHOLDS:
        return "—"
    t = RATIO_THRESHOLDS[key]
    higher = t["higher_is_better"]
    green, amber = t["green"], t["amber"]
    if key in _PERCENT_RATIOS:
        g_s, a_s = f"{green * 100:.0f}%", f"{amber * 100:.0f}%"
    elif key in _DAYS_RATIOS:
        g_s, a_s = f"{green:.0f}d", f"{amber:.0f}d"
    else:
        g_s, a_s = f"{green:.2f}", f"{amber:.2f}"
    return f"Green {'≥' if higher else '≤'}{g_s} / Amber {'≥' if higher else '≤'}{a_s}"


def _status_bg(status: str):
    return {"GREEN": _GREEN, "AMBER": _AMBER, "RED": _RED}.get(status)


class CreditMemoGenerator:
    """Generate a formatted PDF credit memo for a given company's financial data."""

    def __init__(
        self,
        company_name: str,
        fs: FinancialStatements,
        analyst_name: str = "Caleb Mugambi",
        analyst_title: str = "Fintech & Credit Analysis",
        model_path: str = "models/credit_model.pkl",
    ):
        self.company_name = company_name
        self.fs = fs
        self.analyst_name = analyst_name
        self.analyst_title = analyst_title
        self.analysis_date = date.today().strftime("%d %B %Y")
        self.analysis = generate_combined_recommendation(fs, model_path=model_path)
        raw = f"{company_name}{self.analysis_date}".encode()
        self.doc_id = hashlib.sha256(raw).hexdigest()[:8].upper()

    # ── Style factory ──────────────────────────────────────────────────────────

    def _styles(self) -> dict:
        def ps(name, **kw) -> ParagraphStyle:
            return ParagraphStyle(name, **kw)

        return {
            "MemoTitle": ps(
                "MemoTitle", fontName="Helvetica-Bold", fontSize=24,
                textColor=_NAVY, spaceAfter=12, alignment=TA_CENTER,
            ),
            "MemoSubtitle": ps(
                "MemoSubtitle", fontName="Helvetica-Bold", fontSize=16,
                textColor=_NAVY, spaceAfter=8, alignment=TA_CENTER,
            ),
            "SectionHeading": ps(
                "SectionHeading", fontName="Helvetica-Bold", fontSize=16,
                textColor=_NAVY, spaceBefore=14, spaceAfter=8,
            ),
            "SubHeading": ps(
                "SubHeading", fontName="Helvetica-Bold", fontSize=12,
                textColor=_NAVY, spaceBefore=10, spaceAfter=4,
            ),
            "Body": ps(
                "Body", fontName="Helvetica", fontSize=10,
                textColor=_DKGREY, spaceAfter=6, leading=14,
            ),
            "BodyItalic": ps(
                "BodyItalic", fontName="Helvetica-Oblique", fontSize=10,
                textColor=_DKGREY, spaceAfter=6, leading=14,
            ),
            "Recommendation": ps(
                "Recommendation", fontName="Helvetica-Bold", fontSize=20,
                textColor=_NAVY, spaceAfter=10, alignment=TA_CENTER,
            ),
            "CoverMeta": ps(
                "CoverMeta", fontName="Helvetica", fontSize=11,
                textColor=_DKGREY, spaceAfter=6, alignment=TA_CENTER,
            ),
            "Confidential": ps(
                "Confidential", fontName="Helvetica-Oblique", fontSize=9,
                textColor=colors.HexColor("#888888"), alignment=TA_CENTER,
            ),
        }

    # ── Section builders ───────────────────────────────────────────────────────

    def _cover_page(self, st: dict) -> list:
        elems = [Spacer(1, 4 * cm)]
        elems.append(Paragraph("CORPORATE CREDIT MEMO", st["MemoTitle"]))
        elems.append(Spacer(1, 0.5 * cm))
        elems.append(Paragraph(self.company_name, st["MemoSubtitle"]))
        elems.append(Spacer(1, 1.5 * cm))

        meta = Table(
            [
                ["Analysis Date:", self.analysis_date],
                ["Analyst Name:", self.analyst_name],
                ["Analyst Title:", self.analyst_title],
                ["Document ID:", self.doc_id],
            ],
            colWidths=[5 * cm, 9 * cm],
        )
        meta.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("TEXTCOLOR", (0, 0), (-1, -1), _DKGREY),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LGREY, _WHITE]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]))
        elems.append(meta)
        elems.append(Spacer(1, 5 * cm))
        elems.append(Paragraph(
            "CONFIDENTIAL — For internal credit review purposes only",
            st["Confidential"],
        ))
        elems.append(PageBreak())
        return elems

    def _executive_summary(self, st: dict) -> list:
        rb = self.analysis["rules_based"]
        ml = self.analysis["ml_based"]
        grade = rb["letter_grade"]
        rec = self.analysis["final_recommendation"]

        elems = [Paragraph("Executive Summary", st["SectionHeading"]), Spacer(1, 0.3 * cm)]

        summary = Table(
            [
                ["Metric", "Value"],
                ["Letter Grade", grade],
                ["Total Score", f"{rb['total_score']:.1f} / 100"],
                ["Probability of Default", f"{ml['probability_of_default']:.1%}"],
                ["Risk Band", ml["risk_band"]],
            ],
            colWidths=[8 * cm, 8 * cm],
        )
        summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LGREY, _WHITE]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (1, 1), (1, 1), _GRADE_COLORS.get(grade, _WHITE)),
            ("BACKGROUND", (1, 4), (1, 4), _RISK_BAND_COLORS.get(ml["risk_band"], _WHITE)),
        ]))
        elems.append(summary)
        elems.append(Spacer(1, 0.5 * cm))

        rec_bg = _GREEN if rec == "Approve" else _AMBER if rec == "Approve with Conditions" else _RED
        callout = Table(
            [[Paragraph(f"Recommendation: {rec}", st["Recommendation"])]],
            colWidths=[16 * cm],
        )
        callout.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), rec_bg),
            ("BOX", (0, 0), (-1, -1), 1, _NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        elems.append(callout)
        elems.append(Spacer(1, 0.4 * cm))
        elems.append(Paragraph(self.analysis["rationale"], st["Body"]))

        if self.analysis["warnings"]:
            warn_text = " | ".join(self.analysis["warnings"])
            elems.append(Paragraph(f"<i>Warnings: {warn_text}</i>", st["BodyItalic"]))

        elems.append(PageBreak())
        return elems

    def _financial_analysis(self, st: dict) -> list:
        elems = [Paragraph("Financial Analysis", st["SectionHeading"])]
        breakdown = self.analysis["rules_based"]["ratio_breakdown"]
        ratios = self.analysis["ratios"]

        for cat_name, keys in _CATEGORIES.items():
            elems.append(Paragraph(f"{cat_name} Ratios", st["SubHeading"]))
            rows = [["Ratio", "Value", "Status", "Threshold"]]
            status_colors: dict[int, object] = {}

            for i, key in enumerate(keys, start=1):
                value = ratios.get(key, float("nan"))
                val_str = _fmt_value(key, value)

                if key in breakdown:
                    status = breakdown[key]["status"]
                else:
                    status = "N/A" if _is_nan(value) else "—"

                rows.append([_RATIO_LABELS.get(key, key), val_str, status, _fmt_threshold(key)])
                bg = _status_bg(status)
                if bg is not None:
                    status_colors[i] = bg

            tbl = Table(rows, colWidths=[5.5 * cm, 3.5 * cm, 2.5 * cm, 5 * cm])
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (3, 0), (3, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LGREY, _WHITE]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
            for row_i, bg in status_colors.items():
                style_cmds.append(("BACKGROUND", (2, row_i), (2, row_i), bg))
            tbl.setStyle(TableStyle(style_cmds))
            elems.append(tbl)
            elems.append(Spacer(1, 0.4 * cm))

        elems.append(PageBreak())
        return elems

    def _risk_assessment(self, st: dict) -> list:
        elems = [Paragraph("Risk Assessment", st["SectionHeading"])]
        ratios = self.analysis["ratios"]
        breakdown = self.analysis["rules_based"]["ratio_breakdown"]

        # Altman Z-Score
        elems.append(Paragraph("Altman Z-Score Analysis", st["SubHeading"]))
        z = ratios.get("solvency.altman_z_score", float("nan"))
        if _is_nan(z):
            zone, interp = "Unknown", "Insufficient data to compute Altman Z-Score."
            z_bg = _WHITE
        elif z > 2.99:
            zone = "Safe Zone"
            interp = (f"Z-Score of {z:.2f} indicates the company is in the Safe Zone "
                      f"(Z > 2.99), suggesting low bankruptcy risk.")
            z_bg = _GREEN
        elif z >= 1.81:
            zone = "Grey Zone"
            interp = (f"Z-Score of {z:.2f} places the company in the Grey Zone "
                      f"(1.81–2.99), indicating moderate uncertainty.")
            z_bg = _AMBER
        else:
            zone = "Distress Zone"
            interp = (f"Z-Score of {z:.2f} places the company in the Distress Zone "
                      f"(Z < 1.81), signalling elevated bankruptcy risk.")
            z_bg = _RED

        z_tbl = Table(
            [["Z-Score Value", "Zone"], [f"{z:.2f}" if not _is_nan(z) else "N/A", zone]],
            colWidths=[8 * cm, 8 * cm],
        )
        z_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 1), (-1, 1), z_bg),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
        ]))
        elems.append(z_tbl)
        elems.append(Spacer(1, 0.2 * cm))
        elems.append(Paragraph(interp, st["Body"]))
        elems.append(Spacer(1, 0.3 * cm))

        # Key risk factors
        elems.append(Paragraph("Key Risk Factors", st["SubHeading"]))
        red_items = [(k, v) for k, v in breakdown.items() if v["status"] == "RED"]
        if red_items:
            for key, info in red_items:
                label = _RATIO_LABELS.get(key, key)
                elems.append(Paragraph(
                    f"• {label} is below acceptable thresholds at {_fmt_value(key, info['value'])}.",
                    st["Body"],
                ))
        else:
            elems.append(Paragraph(
                "No material risk factors identified from quantitative analysis.",
                st["Body"],
            ))

        elems.append(Spacer(1, 0.3 * cm))

        # Model confidence
        elems.append(Paragraph("Model Confidence", st["SubHeading"]))
        conf = self.analysis["ml_based"]["model_confidence"]
        elems.append(Paragraph(f"Confidence: {conf}", st["Body"]))
        if conf == "Reduced":
            elems.append(Paragraph(
                "ML model confidence is Reduced due to missing input features; "
                "ML signals should be weighted less in the final decision.",
                st["BodyItalic"],
            ))

        elems.append(PageBreak())
        return elems

    def _recommendation_page(self, st: dict) -> list:
        rec = self.analysis["final_recommendation"]
        elems = [
            Paragraph("Recommendation", st["SectionHeading"]),
            Spacer(1, 0.5 * cm),
            Paragraph(rec, st["Recommendation"]),
            Spacer(1, 0.5 * cm),
        ]

        if rec == "Approve with Conditions":
            elems.append(Paragraph("Conditions", st["SubHeading"]))
            for cond in [
                "• Minimum current ratio of 1.2x to be maintained at all times.",
                "• Maximum debt-to-equity ratio of 2.5x.",
                "• Quarterly financial reporting to be submitted within 45 days of period end.",
                "• Material adverse change clause to be included in the facility agreement.",
            ]:
                elems.append(Paragraph(cond, st["Body"]))
            elems.append(Spacer(1, 0.5 * cm))

        elems.append(Spacer(1, 1 * cm))
        signoff = Table(
            [
                ["Prepared by:", self.analyst_name],
                ["Title:", self.analyst_title],
                ["Date:", self.analysis_date],
            ],
            colWidths=[4 * cm, 10 * cm],
        )
        signoff.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, -1), _DKGREY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, _NAVY),
        ]))
        elems.append(signoff)
        return elems

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, output_path: str) -> str:
        """Build the PDF at output_path and return its absolute path."""
        abs_path = os.path.abspath(output_path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        doc = SimpleDocTemplate(
            abs_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        st = self._styles()
        story = []
        story.extend(self._cover_page(st))
        story.extend(self._executive_summary(st))
        story.extend(self._financial_analysis(st))
        story.extend(self._risk_assessment(st))
        story.extend(self._recommendation_page(st))
        doc.build(story)
        return abs_path
