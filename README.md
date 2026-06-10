# Credit Analysis Automator

A corporate credit analysis tool that streamlines the end-to-end credit assessment workflow. It ingests financial statements, computes key financial ratios, applies a machine-learning credit-scoring model, generates professional PDF credit memos, and surfaces an interactive risk dashboard — enabling analysts to produce consistent, data-driven credit opinions faster and with greater accuracy.

## Features

- **Financial Ratios** — Automated computation of liquidity, leverage, profitability, and coverage ratios from raw financial data
- **ML Credit Scoring** — Scikit-learn powered model that predicts creditworthiness and assigns a risk grade
- **PDF Credit Memo** — ReportLab-generated, board-ready credit memoranda with ratio tables, charts, and recommendations
- **Risk Dashboard** — Interactive Streamlit + Plotly dashboard for portfolio-level risk monitoring and drill-down analysis

## Tech Stack

| Layer | Library |
|---|---|
| Dashboard / UI | Streamlit, Plotly |
| Data Processing | Pandas, NumPy, OpenPyXL |
| ML Scoring | Scikit-learn, Joblib |
| PDF Generation | ReportLab |
| Testing | Pytest |

## Project Structure

```
credit-analysis-automator/
├── src/
│   ├── __init__.py
│   └── scoring/
│       └── __init__.py
├── data/             # Raw and processed financial data
├── models/           # Serialised ML model artefacts
├── notebooks/        # Exploratory analysis notebooks
├── tests/
│   └── __init__.py
├── docs/             # Technical documentation
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd credit-analysis-automator

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Launch the risk dashboard
streamlit run src/app.py
```

> Full usage documentation coming soon.

---

Built by Caleb Mugambi — Fintech & Credit Analysis
