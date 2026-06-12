"""Credit Analysis Automator — Streamlit web application."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ratios import FinancialStatements
from src.scoring.combined import generate_combined_recommendation
from src.scoring.ml_model import load_credit_model
from src.memo_generator import CreditMemoGenerator
from src.app_helpers import (
    CATEGORIES,
    DAYS_RATIOS,
    PERCENT_RATIOS,
    RATIO_LABELS,
    financial_statements_from_upload,
    format_ratio_value,
    get_sample_financials,
    get_template_dataframe,
    status_color,
)

# ── Page config (must be the very first Streamlit call) ───────────────────────

st.set_page_config(
    page_title="Credit Analysis Automator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached model loader ────────────────────────────────────────────────────────

@st.cache_resource
def _load_model(model_path: str = "models/credit_model.pkl"):
    return load_credit_model(model_path)


# ── Threshold formatter (local — used only in the ratio table) ────────────────

def _fmt_threshold(key: str) -> str:
    from src.scoring.rules_engine import RATIO_THRESHOLDS
    if key not in RATIO_THRESHOLDS:
        return "—"
    t = RATIO_THRESHOLDS[key]
    higher = t["higher_is_better"]
    green, amber = t["green"], t["amber"]
    if key in PERCENT_RATIOS:
        g_s, a_s = f"{green * 100:.0f}%", f"{amber * 100:.0f}%"
    elif key in DAYS_RATIOS:
        g_s, a_s = f"{green:.0f}d", f"{amber:.0f}d"
    else:
        g_s, a_s = f"{green:.2f}", f"{amber:.2f}"
    sym = "≥" if higher else "≤"
    return f"Green {sym}{g_s} / Amber {sym}{a_s}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This tool automates corporate credit analysis using a **rules-based + ML hybrid** engine. "
        "It computes 19 financial ratios (including the **Altman Z-Score**), applies a trained "
        "logistic-regression model to estimate probability of default, and produces a "
        "**downloadable PDF credit memo**."
    )

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("- Python\n- Streamlit\n- scikit-learn\n- ReportLab\n- Plotly")

    with st.expander("Methodology"):
        st.markdown(
            "The scoring engine combines two complementary approaches. "
            "A **rules-based scorecard** classifies each of nine key ratios as GREEN, AMBER, or RED "
            "against industry-standard thresholds, producing a 0–100 score that maps to a letter "
            "grade (AAA–D). An **ML logistic-regression model** then estimates the probability of "
            "default from all 19 computed ratios. The final decision integrates both signals: "
            "unanimous positives yield *Approve*, divergence yields *Approve with Conditions*, "
            "and strong negatives yield *Decline*."
        )

    st.markdown("---")
    st.markdown(
        "Built by **Caleb Mugambi**  \n"
        "[GitHub](https://github.com/yourusername) · [LinkedIn](#)"
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Credit Analysis Automator")
st.caption("Corporate credit analysis in seconds")
st.markdown(
    '<p style="color:gray;font-size:0.85em;">by Caleb Mugambi — Fintech &amp; Credit Analysis</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Company name ──────────────────────────────────────────────────────────────

company_name = st.text_input("Company Name", value="Sample Company Ltd")

# ── Input section ─────────────────────────────────────────────────────────────

st.subheader("1. Provide Company Financials")
input_mode = st.radio(
    "Data source",
    [
        "Use Sample Data (Green Solutions Manufacturing Ltd)",
        "Upload Excel/CSV",
        "Enter Manually",
    ],
    label_visibility="collapsed",
)

fs: FinancialStatements | None = None

# ── Sample data ────────────────────────────────────────────────────────────────
if input_mode == "Use Sample Data (Green Solutions Manufacturing Ltd)":
    fs = get_sample_financials()
    st.session_state["fs"] = fs
    with st.expander("View sample financials"):
        sample_df = get_template_dataframe()
        st.dataframe(
            sample_df.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

# ── Upload ────────────────────────────────────────────────────────────────────
elif input_mode == "Upload Excel/CSV":
    st.markdown(
        "**Expected columns** — one column per `FinancialStatements` field:  \n"
        "`current_assets`, `cash`, `inventory`, `receivables`, `total_assets`, "
        "`current_liabilities`, `total_liabilities`, `total_equity`, `retained_earnings`, "
        "`working_capital`, `revenue`, `cogs`, `ebit`, `ebitda`, `interest_expense`, "
        "`net_income`, `tax_expense`, `operating_cash_flow`, `capex`, `market_cap`"
    )
    template_csv = get_template_dataframe().to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_csv,
        file_name="template_financials.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload your financials", type=["xlsx", "csv"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded)
            else:
                df_upload = pd.read_excel(uploaded)
            fs = financial_statements_from_upload(df_upload)
            st.session_state["fs"] = fs
            st.success("File loaded successfully.")
        except Exception as exc:
            st.error(f"Error reading file: {exc}")

# ── Manual entry ──────────────────────────────────────────────────────────────
elif input_mode == "Enter Manually":
    with st.form("financials_form"):
        st.markdown("Fill in the financial data below (leave 0 for unknown values).")

        with st.expander("Balance Sheet", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                v_current_assets      = st.number_input("Current Assets",          value=0.0, help="Total short-term assets")
                v_cash                = st.number_input("Cash & Equivalents",       value=0.0, help="Cash and liquid equivalents")
                v_inventory           = st.number_input("Inventory",                value=0.0, help="Inventory value")
                v_receivables         = st.number_input("Accounts Receivable",      value=0.0, help="Trade receivables")
                v_total_assets        = st.number_input("Total Assets",             value=0.0, help="Total balance sheet assets")
            with c2:
                v_current_liabilities = st.number_input("Current Liabilities",      value=0.0, help="Short-term obligations")
                v_total_liabilities   = st.number_input("Total Liabilities",        value=0.0, help="All liabilities")
                v_total_equity        = st.number_input("Total Equity",             value=0.0, help="Shareholders' equity")
                v_retained_earnings   = st.number_input("Retained Earnings",        value=0.0, help="Cumulative retained earnings (can be negative)")
                v_market_cap          = st.number_input("Market Capitalisation",    value=0.0, help="Market cap (needed for Altman Z-Score)")

        with st.expander("Income Statement", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                v_revenue          = st.number_input("Revenue",           value=0.0, help="Total sales / revenue")
                v_cogs             = st.number_input("COGS",              value=0.0, help="Cost of goods sold")
                v_ebit             = st.number_input("EBIT",              value=0.0, help="Earnings before interest and tax")
                v_ebitda           = st.number_input("EBITDA",            value=0.0, help="Earnings before interest, tax, D&A")
            with c2:
                v_interest_expense = st.number_input("Interest Expense",  value=0.0, help="Annual interest charges")
                v_net_income       = st.number_input("Net Income",         value=0.0, help="Bottom-line net profit")
                v_tax_expense      = st.number_input("Tax Expense",        value=0.0, help="Income tax expense")

        with st.expander("Cash Flow", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                v_operating_cash_flow = st.number_input("Operating Cash Flow", value=0.0, help="Cash generated from operations")
            with c2:
                v_capex = st.number_input("Capital Expenditure (CapEx)", value=0.0, help="Capital spending")

        submitted = st.form_submit_button("Save Financials")

    if submitted:
        def _z(v: float) -> float | None:
            return None if v == 0.0 else v

        fs = FinancialStatements(
            current_assets=_z(v_current_assets),
            cash=_z(v_cash),
            inventory=_z(v_inventory),
            receivables=_z(v_receivables),
            total_assets=_z(v_total_assets),
            current_liabilities=_z(v_current_liabilities),
            total_liabilities=_z(v_total_liabilities),
            total_equity=_z(v_total_equity),
            retained_earnings=_z(v_retained_earnings),
            revenue=_z(v_revenue),
            cogs=_z(v_cogs),
            ebit=_z(v_ebit),
            ebitda=_z(v_ebitda),
            interest_expense=_z(v_interest_expense),
            net_income=_z(v_net_income),
            tax_expense=_z(v_tax_expense),
            operating_cash_flow=_z(v_operating_cash_flow),
            capex=_z(v_capex),
            market_cap=_z(v_market_cap),
        )
        st.session_state["fs"] = fs
        st.success("Financials saved. Click 'Run Credit Analysis' below.")

# Restore fs from session state if not set by the current mode
if fs is None and "fs" in st.session_state:
    fs = st.session_state["fs"]

# ── Analyze button ────────────────────────────────────────────────────────────

st.divider()
if st.button("🚀 Run Credit Analysis", type="primary", disabled=(fs is None)):
    with st.spinner("Running analysis…"):
        analysis = generate_combined_recommendation(fs)
        st.session_state["analysis"] = analysis
        st.session_state["snap_company"] = company_name
        st.session_state["snap_fs"] = fs

# ── Results ───────────────────────────────────────────────────────────────────

if "analysis" in st.session_state:
    analysis    = st.session_state["analysis"]
    snap_name   = st.session_state.get("snap_company", company_name)
    snap_fs     = st.session_state.get("snap_fs", fs)

    rb  = analysis["rules_based"]
    ml  = analysis["ml_based"]
    rec = analysis["final_recommendation"]

    grade    = rb["letter_grade"]
    pd_score = ml["probability_of_default"]
    ratios   = analysis["ratios"]
    breakdown = rb["ratio_breakdown"]

    st.subheader("2. Results")

    # KPI row ─────────────────────────────────────────────────────────────────
    rec_emoji = {"Approve": "✅", "Approve with Conditions": "⚠️", "Decline": "❌"}.get(rec, "")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Letter Grade", grade)
    with col2:
        delta_pp = (pd_score - 0.25) * 100
        st.metric(
            "Probability of Default",
            f"{pd_score:.1%}",
            delta=f"{delta_pp:+.1f}pp vs 25% baseline",
            delta_color="inverse",
        )
    with col3:
        st.metric("Final Recommendation", f"{rec_emoji} {rec}")

    # Tabs ────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ratios", "Risk Visualizations", "Detailed Analysis", "Warnings"]
    )

    # ── TAB 1: Ratios ─────────────────────────────────────────────────────────
    with tab1:
        rows = []
        for cat, keys in CATEGORIES.items():
            for key in keys:
                val = ratios.get(key, float("nan"))
                status = breakdown.get(key, {}).get("status", "—")
                rows.append({
                    "Category":  cat,
                    "Ratio":     RATIO_LABELS.get(key, key),
                    "Value":     format_ratio_value(key, val),
                    "Status":    status,
                    "Threshold": _fmt_threshold(key),
                })

        ratio_df = pd.DataFrame(rows)

        def _cell_color(val: str) -> str:
            bg = {"GREEN": "#c8f7c5", "AMBER": "#fef3c7", "RED": "#fecaca"}.get(val, "")
            return f"background-color: {bg}" if bg else ""

        styled = ratio_df.style.map(_cell_color, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── TAB 2: Risk Visualizations ────────────────────────────────────────────
    with tab2:
        # Horizontal bar chart
        bd_keys     = list(breakdown.keys())
        bd_labels   = [RATIO_LABELS.get(k, k) for k in bd_keys]
        bd_scores   = [breakdown[k]["points"] for k in bd_keys]
        bd_statuses = [breakdown[k]["status"] for k in bd_keys]
        bd_colors   = [status_color(s) for s in bd_statuses]

        bar_fig = go.Figure(go.Bar(
            y=bd_labels,
            x=bd_scores,
            orientation="h",
            marker_color=bd_colors,
            text=[f"{p:.1f}pts" for p in bd_scores],
            textposition="outside",
        ))
        bar_fig.update_layout(
            title="Ratio Score Breakdown",
            xaxis_title="Points",
            height=420,
            margin=dict(l=210),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Gauge + Z-Score side by side
        col_g, col_z = st.columns(2)

        with col_g:
            total_score = rb["total_score"]
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=total_score,
                title={"text": "Total Score (0–100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#1a365d"},
                    "steps": [
                        {"range": [0,  40],  "color": "#FF7F7F"},
                        {"range": [40, 65],  "color": "#FFD580"},
                        {"range": [65, 100], "color": "#90EE90"},
                    ],
                },
            ))
            gauge_fig.update_layout(height=350)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_z:
            z = ratios.get("solvency.altman_z_score", float("nan"))
            z_valid = not (isinstance(z, float) and math.isnan(z))
            if z_valid:
                z_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=z,
                    delta={"reference": 2.99},
                    title={"text": "Altman Z-Score"},
                    gauge={
                        "axis": {"range": [0, 6]},
                        "bar":  {"color": "#1a365d"},
                        "steps": [
                            {"range": [0,    1.81], "color": "#FF7F7F"},
                            {"range": [1.81, 2.99], "color": "#FFD580"},
                            {"range": [2.99, 6],    "color": "#90EE90"},
                        ],
                        "threshold": {
                            "line": {"color": "navy", "width": 4},
                            "thickness": 0.75,
                            "value": 2.99,
                        },
                    },
                ))
                z_fig.update_layout(height=350)
                st.plotly_chart(z_fig, use_container_width=True)
            else:
                st.info("Altman Z-Score unavailable — missing market or financial data.")

    # ── TAB 3: Detailed Analysis ──────────────────────────────────────────────
    with tab3:
        st.markdown(f"**Rationale:** {analysis['rationale']}")
        st.markdown(f"**Model Confidence:** {ml['model_confidence']}")

        if rec == "Approve with Conditions":
            st.markdown("**Conditions:**")
            for cond in [
                "Minimum current ratio of 1.2x to be maintained at all times.",
                "Maximum debt-to-equity ratio of 2.5x.",
                "Quarterly financial reporting to be submitted within 45 days of period end.",
                "Material adverse change clause to be included in the facility agreement.",
            ]:
                st.markdown(f"- {cond}")

    # ── TAB 4: Warnings ───────────────────────────────────────────────────────
    with tab4:
        warnings = analysis.get("warnings", [])
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("✅ No warnings flagged.")

    # ── PDF download ──────────────────────────────────────────────────────────
    st.divider()
    if st.button("📄 Generate & Download PDF Credit Memo"):
        with st.spinner("Generating PDF…"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            CreditMemoGenerator(snap_name, snap_fs).generate(tmp_path)
            with open(tmp_path, "rb") as fh:
                pdf_bytes = fh.read()
            os.unlink(tmp_path)

        today     = date.today().strftime("%Y%m%d")
        safe_name = snap_name.replace(" ", "_")
        st.download_button(
            "📥 Download PDF",
            data=pdf_bytes,
            file_name=f"credit_memo_{safe_name}_{today}.pdf",
            mime="application/pdf",
        )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:gray;font-size:0.8em;">'
    "Credit Analysis Automator v0.1 — Built by Caleb Mugambi — Fintech &amp; Credit Analysis"
    "</p>",
    unsafe_allow_html=True,
)
