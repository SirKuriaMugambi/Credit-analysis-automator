# Credit Analysis Automator — Methodology

## Overview

This tool implements a **hybrid credit scoring approach** that mirrors real-world credit risk practice: a transparent rules-based scorecard provides an explainable, auditable decision trail, while a machine learning model adds statistical depth by estimating the probability of default from the full ratio profile. Both signals are cross-referenced to produce a final recommendation, with disagreements surfaced as analyst warnings rather than silently overridden.

---

## Financial Ratios

The engine computes 19 financial ratios grouped into six categories.

### Liquidity

| Ratio | Formula | What It Measures |
|---|---|---|
| Current Ratio | Current Assets / Current Liabilities | Ability to cover short-term obligations |
| Quick Ratio | (Current Assets − Inventory) / Current Liabilities | Liquidity excluding slow-moving inventory |
| Cash Ratio | Cash / Current Liabilities | Most conservative liquidity measure |

### Leverage

| Ratio | Formula | What It Measures |
|---|---|---|
| Debt to Equity | Total Liabilities / Total Equity | Financial leverage and creditor risk |
| Debt to Assets | Total Liabilities / Total Assets | Proportion of assets financed by debt |
| Equity Multiplier | Total Assets / Total Equity | Financial leverage (DuPont component) |
| Interest Coverage | EBIT / Interest Expense | Ability to service interest payments |

### Profitability

| Ratio | Formula | What It Measures |
|---|---|---|
| Gross Margin | (Revenue − COGS) / Revenue | Core production profitability |
| EBITDA Margin | EBITDA / Revenue | Operating profitability before D&A |
| Net Margin | Net Income / Revenue | Bottom-line profitability |
| Return on Assets (ROA) | Net Income / Total Assets | Asset utilisation efficiency |
| Return on Equity (ROE) | Net Income / Total Equity | Return generated for shareholders |

### Efficiency

| Ratio | Formula | What It Measures |
|---|---|---|
| Asset Turnover | Revenue / Total Assets | Revenue generated per unit of assets |
| Days Sales Outstanding (DSO) | (Receivables / Revenue) × 365 | Average days to collect receivables |
| Days Inventory Outstanding (DIO) | (Inventory / COGS) × 365 | Average days to sell inventory |
| Cash Conversion Cycle | DSO + DIO | Total days to convert inputs to cash |

### Cash Flow

| Ratio | Formula | What It Measures |
|---|---|---|
| Free Cash Flow | Operating Cash Flow − CapEx | Cash available after capital expenditure |
| FCF to Debt | Free Cash Flow / Total Liabilities | Ability to repay debt from free cash flow |

### Solvency

| Ratio | Formula | What It Measures |
|---|---|---|
| Altman Z-Score | See below | Overall bankruptcy distress indicator |

---

## Altman Z-Score

The Altman Z-Score uses five financial ratios to predict corporate bankruptcy risk:

```
Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
```

| Variable | Definition |
|---|---|
| A | Working Capital / Total Assets |
| B | Retained Earnings / Total Assets |
| C | EBIT / Total Assets |
| D | Market Capitalisation / Total Liabilities |
| E | Revenue / Total Assets |

### Distress Zones

| Z-Score | Zone | Interpretation |
|---|---|---|
| > 2.99 | **Safe** | Low bankruptcy risk |
| 1.81 – 2.99 | **Grey** | Uncertain — warrants closer monitoring |
| < 1.81 | **Distress** | High bankruptcy risk |

*Reference: Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." Journal of Finance.*

---

## Rules-Based Scorecard

Nine key ratios are scored against industry-standard thresholds using a traffic-light system.

### Threshold Table

| Ratio | Green (full points) | Amber (half points) | Direction |
|---|---|---|---|
| Current Ratio | ≥ 2.0 | ≥ 1.2 | Higher is better |
| Quick Ratio | ≥ 1.0 | ≥ 0.7 | Higher is better |
| Debt to Equity | ≤ 1.5 | ≤ 2.5 | Lower is better |
| Interest Coverage | ≥ 3.0 | ≥ 1.5 | Higher is better |
| EBITDA Margin | ≥ 20% | ≥ 10% | Higher is better |
| Net Margin | ≥ 10% | ≥ 5% | Higher is better |
| Return on Assets | ≥ 10% | ≥ 5% | Higher is better |
| FCF to Debt | ≥ 20% | ≥ 10% | Higher is better |
| Debt to Assets | ≤ 0.40 | ≤ 0.60 | Lower is better |

### Scoring Rules

- **GREEN** → Full points awarded
- **AMBER** → Half points awarded
- **RED** → Zero points awarded
- **NaN / missing** → Zero points awarded

### Score-to-Grade Mapping

| Score Range | Letter Grade | Decision |
|---|---|---|
| 85 – 100 | AAA | Approve |
| 75 – 84 | AA | Approve |
| 65 – 74 | A | Approve |
| 55 – 64 | BBB | Approve with Conditions |
| 45 – 54 | BB | Approve with Conditions |
| 35 – 44 | B | Approve with Conditions |
| 25 – 34 | CCC | Decline |
| 15 – 24 | CC | Decline |
| 0 – 14 | D | Decline |

---

## ML Model

The ML component is a scikit-learn pipeline:

```
SimpleImputer(strategy="median") → StandardScaler → LogisticRegression(max_iter=1000)
```

- **Input features**: All 19 computed ratios
- **Training data**: Synthetic dataset with a 70/30 healthy/distressed split, calibrated to reflect Altman's distressed-firm financial characteristics
- **Output**: Probability of default (PD) in the range [0, 1]

> **Limitation**: The model is trained on synthetic data. For production use, retrain on a vetted historical bankruptcy dataset and validate against your institution's risk policies.

---

## Combined Recommendation Logic

The final decision cross-references both scoring signals:

| Rules Score | ML Probability of Default | Recommendation |
|---|---|---|
| Grade ≥ BBB (approve) | PD < 0.30 | **Approve** |
| Grade < CCC (decline) OR | PD > 0.60 | **Decline** |
| All other combinations | — | **Approve with Conditions** |

When the two signals disagree — for example, a rules-based approval but ML PD > 0.5 — a warning is surfaced for analyst review rather than silently overriding either signal.

---

## Risk Bands

| Probability of Default | Risk Band |
|---|---|
| PD < 0.20 | Low |
| 0.20 ≤ PD ≤ 0.50 | Medium |
| PD > 0.50 | High |

---

## Limitations & Future Work

- **Synthetic training data**: The ML model has not been trained on real historical bankruptcy data. Real-world performance may differ significantly.
- **No industry benchmarks**: Thresholds are generic. Different industries (e.g., retail vs. capital-intensive manufacturing) have materially different norms.
- **No macroeconomic factors**: Interest rate cycles, GDP growth, and sector-level stress are not incorporated.
- **Model retraining cadence**: The model should be retrained periodically as economic conditions evolve.
- **Single-period analysis**: The tool analyses a single set of financials; trend analysis across multiple periods would add predictive power.

---

## References

- Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *Journal of Finance*, 23(4), 589–609.
- scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
