# 🔍 Insurance Claim Settlement Bias Analyser

A Streamlit dashboard for detecting bias in insurance claim settlements through descriptive, diagnostic, and predictive analytics.

---

## 🚀 Live Demo

Deploy on **Streamlit Community Cloud** (free) in under 5 minutes:

1. Fork / push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → connect your repo → set `app.py` as the main file
4. Click **Deploy** ✅

---

## 📁 Project Structure

```
insurance_bias_dashboard/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── generate_sample_data.py ← (Optional) generate a synthetic dataset
├── Insurance.csv           ← Your dataset (upload via sidebar in the app)
└── README.md
```

---

## 📊 Features

### Tab 1 — Descriptive Analysis
- Interactive cross-tabulation against Policy Status for any dimension
- Chi-Square test for statistical significance
- Univariate distribution plots (age, income, sum assured, gender, payment mode)

### Tab 2 — Diagnostic Bias Analysis
- **Age-wise bias**: Histograms, boxplots, independent t-test
- **Income-wise bias**: Income group approval rates, boxplots, t-test
- **Zone/Team-wise bias**: Horizontal bar chart of approval rates per zone, Chi-Square
- **Gender × Age interaction**: Heatmap of approval rates
- **Correlation heatmap**: Numeric features vs outcome

### Tab 3 — Feature Engineering
- Numeric cleaning (comma removal, casting)
- Missing value imputation
- Label encoding for categorical variables
- Log transforms (income, sum assured)
- Engineered ratio feature
- StandardScaler + stratified train/test split

### Tab 4 — ML Models
- **KNN** (k=7)
- **Decision Tree** (max_depth=6)
- **Random Forest** (100 estimators)
- **Gradient Boosting** (100 estimators)
- Feature importance charts
- Detailed classification reports per model

### Tab 5 — Model Performance
- Train vs Test accuracy comparison
- Precision / Recall / F1 line chart
- ROC curves for all models on a single plot
- Confusion matrices for all 4 models

### Tab 6 — Findings & Recommendations
- Dynamic, data-driven findings (age, income, zone, medical, early-claim bias)
- 6 actionable recommendations for the claims management team

---

## ⚙️ Local Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/insurance-bias-dashboard.git
cd insurance-bias-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Upload your `Insurance.csv` via the sidebar. The CSV should contain these columns:

| Column | Description |
|---|---|
| POLICY_NO | Policy number |
| PI_GENDER | M / F |
| SUM_ASSURED | Sum assured (can have commas) |
| ZONE | Settlement zone / team |
| PAYMENT_MODE | Annual / Half-Yly / Quarterly / Monthly / Single |
| EARLY_NON | EARLY / NON EARLY |
| PI_OCCUPATION | Occupation string |
| MEDICAL_NONMED | MEDICAL / NON MEDICAL |
| PI_STATE | Indian state |
| REASON_FOR_CLAIM | Cause of claim (can be blank) |
| PI_AGE | Age in years |
| PI_ANNUAL_INCOME | Annual income (can have commas) |
| POLICY_STATUS | **Approved Death Claim** / **Repudiate Death** |

---

## 📦 Requirements

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

---

## 📝 Disclaimer

This tool is for internal investigative and audit purposes only. Statistical significance does not imply discriminatory intent. All findings should be reviewed with domain experts and legal counsel before formal action.

---

## 🛡️ License

MIT License — free to use and modify for internal compliance and audit work.
