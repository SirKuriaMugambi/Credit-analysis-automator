[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-analysis-automator.streamlit.app)# 📊 Credit Analysis Automator

> Corporate credit analysis in seconds — from raw financial statements to a downloadable PDF credit memo, powered by a hybrid rules + ML scoring engine.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-95%2B%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 🚀 Live Demo

🔗 **[Try the live app](https://credit-analysis-automator.streamlit.app)** *(deployment in progress)*

---

## 📸 Screenshots

*Screenshots will be added once the deployment is finalized.*

<!-- Once available:
![Hero](docs/screenshots/hero.png)
![Risk Dashboard](docs/screenshots/gauges.png)
![Detailed Analysis](docs/screenshots/analysis.png)
-->

---

## ✨ What It Does

- ✅ Computes 19 financial ratios across 6 categories (Liquidity, Leverage, Profitability, Efficiency, Cash Flow, Solvency)
- ✅ Calculates the Altman Z-Score for bankruptcy distress prediction
- ✅ Applies a transparent rules-based scorecard (AAA → D letter grades)
- ✅ Predicts probability of default using a trained scikit-learn ML pipeline
- ✅ Combines both signals into a final Approve / Conditions / Decline recommendation
- ✅ Flags warnings when rules-based and ML scores disagree
- ✅ Generates a downloadable, professional PDF credit memo
- ✅ Interactive Plotly dashboard with gauge charts, ratio breakdowns, and risk visualizations
- ✅ Three input modes: sample data, Excel/CSV upload, manual entry

---

## 💡 Why It Exists

Traditional credit analysis takes hours and the output is buried in spreadsheets. This tool compresses it into seconds with a credit officer's audit trail intact — every score is explainable, every threshold is documented, and the PDF output mirrors the format a real credit committee would review.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Web UI | Streamlit |
| ML / Data | scikit-learn, pandas, numpy |
| Visualization | Plotly |
| PDF Generation | ReportLab |
| Testing | pytest |
| Version Control | Git |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Git

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/SirKuriaMugambi/credit-analysis-automator.git
cd credit-analysis-automator

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
credit-analysis-automator/
├── app.py                      # Streamlit web application
├── requirements.txt
├── README.md
├── LICENSE
├── src/
│   ├── ratios.py               # 19-ratio financial engine + Altman Z
│   ├── memo_generator.py       # PDF credit memo builder (ReportLab)
│   ├── app_helpers.py          # Pure-Python helpers for the Streamlit UI
│   └── scoring/
│       ├── rules_engine.py     # Transparent scorecard (traffic-light)
│       ├── ml_model.py         # scikit-learn logistic regression pipeline
│       └── combined.py         # Hybrid rules + ML recommendation
├── tests/                      # 95+ pytest unit tests
├── data/
│   ├── sample_financials.xlsx  # 3 example companies
│   ├── template_financials.csv # Empty template for user uploads
│   └── sample_memo.pdf         # Example PDF output
├── models/
│   └── credit_model.pkl        # Trained ML pipeline
├── notebooks/
│   └── model_training.ipynb    # EDA, training, evaluation
└── docs/
    └── methodology.md          # Scoring methodology + thresholds
```

---

## 📐 Methodology

The tool combines two complementary scoring approaches:

**Rules-Based Scorecard** — Each ratio is classified Green / Amber / Red against industry-standard thresholds, then aggregated into a 0–100 score and mapped to an AAA → D letter grade. Transparent, explainable, auditable.

**ML Model** — A logistic regression pipeline (Imputer → Scaler → LogisticRegression) trained on synthetic financial data calibrated to Altman's distressed-firm characteristics. Outputs probability of default.

**Combined Recommendation** — The two signals are cross-referenced. Disagreements (e.g., rules approve but ML flags high PD) are surfaced as warnings for analyst review.

📖 [Read the full methodology](docs/methodology.md)

---

## 🧪 Testing

```bash
pytest -v
```

The test suite includes 95+ unit tests across:
- Ratio calculations (healthy / distressed / zero-division edge cases)
- Rules-based scoring (boundary tests for all letter grades)
- ML model training and prediction
- PDF memo generation
- App helper functions

---

## ⚠️ Disclaimer

This tool is for educational and demonstration purposes. The ML model is trained on synthetic data, not real historical bankruptcy data. For production credit decisions, retrain the model on a vetted dataset and validate against your institution's risk policies.

---

## 👤 Author

**Caleb Mugambi**
Fintech & Credit Analysis

🔗 [GitHub](https://github.com/SirKuriaMugambi) · [LinkedIn](https://www.linkedin.com/in/caleb-mugambi)

---

Built with ❤️ in Nairobi.
