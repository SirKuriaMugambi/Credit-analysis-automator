Build a complete Python project scaffold in the current directory. The current directory is already a git repository — do NOT run git init again.

Create these files and folders:

1. requirements.txt with EXACTLY these 9 packages, pinned to recent stable versions:
streamlit, pandas, numpy, scikit-learn, plotly, reportlab, pytest, openpyxl, joblib

2. .gitignore with: venv/, .venv/, __pycache__/, *.pyc, .pytest_cache/, .streamlit/secrets.toml, .DS_Store, *.egg-info/, models/credit_model.pkl

3. Folder structure: src/, src/scoring/, data/, models/, notebooks/, tests/, docs/

4. Empty __init__.py files in: src/, src/scoring/, tests/

5. README.md with:
   - Title: "Credit Analysis Automator"
   - One-paragraph description (corporate credit analysis tool)
   - Features section (financial ratios, ML credit scoring, PDF credit memo, risk dashboard)
   - Tech stack
   - Project structure tree
   - Installation instructions (venv + pip install)
   - Usage placeholder
   - Footer: "Built by Caleb Mugambi — Fintech & Credit Analysis"

6. Stage everything with git and commit: "chore: initial project scaffold"

7. Create a Python virtual environment named .venv

8. Activate the venv and install requirements from requirements.txt

9. Run pip list to verify

Do all file creation in one batch without asking permission between each file. Only stop to ask permission for terminal commands (git, venv creation, pip install).