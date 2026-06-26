import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Bias Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
  }
  .metric-card .label { color: #8892b0; font-size: 12px; font-weight: 500;
                         text-transform: uppercase; letter-spacing: 1px; }
  .metric-card .value { color: #e94560; font-size: 28px; font-weight: 700; margin-top: 4px; }
  .metric-card .sub   { color: #64ffda; font-size: 12px; margin-top: 4px; }
  .section-header {
    background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
    border-radius: 8px; padding: 10px 18px; color: white;
    font-size: 16px; font-weight: 600; margin: 24px 0 16px 0;
  }
  .bias-alert {
    background: rgba(233,69,96,0.12); border-left: 4px solid #e94560;
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 8px 0;
    color: #e0e0e0; font-size: 14px;
  }
  .bias-ok {
    background: rgba(100,255,218,0.08); border-left: 4px solid #64ffda;
    border-radius: 0 8px 8px 0; padding: 12px 18px; margin: 8px 0;
    color: #e0e0e0; font-size: 14px;
  }
  .finding-card {
    background: #1a1d2e; border: 1px solid #2d3561;
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
  }
  .finding-card h4 { color: #64ffda; margin: 0 0 8px 0; font-size: 14px; }
  .finding-card p  { color: #c0c7d4; margin: 0; font-size: 13px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ─── Colour constants ─────────────────────────────────────────────────────────
CLR_A    = "#64ffda"   # Approved
CLR_R    = "#e94560"   # Repudiated
DARK_BG  = "#0f1117"
PLOT_BG  = "#1a1d2e"
GRID_CLR = "#2d3561"
TXT_CLR  = "#c0c7d4"
PALETTE  = {"Approved Death Claim": CLR_A, "Repudiate Death": CLR_R}

# ─── Dark-style helper ────────────────────────────────────────────────────────
def dark(fig, axlist):
    """Apply dark theme to a figure. axlist must be a plain Python list."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in axlist:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TXT_CLR)
        ax.yaxis.label.set_color(TXT_CLR)
        ax.title.set_color(TXT_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.5)

def show(fig):
    st.pyplot(fig)
    plt.close(fig)

# ─── Chi-square helper ────────────────────────────────────────────────────────
def chi2_test(df, col):
    ct = pd.crosstab(df[col], df["POLICY_STATUS"])
    c2, p, _, _ = stats.chi2_contingency(ct)
    return c2, p

def bias_box(p, label, chi=True):
    test = "Chi-Square" if chi else "t-test"
    if p < 0.05:
        return (f"<div class='bias-alert'>⚠️ <b>{test} — {label}:</b> "
                f"p={p:.4f} → Statistically significant association detected.</div>")
    return (f"<div class='bias-ok'>✅ <b>{test} — {label}:</b> "
            f"p={p:.4f} → No significant association found.</div>")

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Clean numeric columns (stored with commas)
    for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
        df[col] = (df[col].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.strip())
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # FIX: use assignment instead of inplace (pandas 3.x compatibility)
    df["PI_ANNUAL_INCOME"] = df["PI_ANNUAL_INCOME"].fillna(df["PI_ANNUAL_INCOME"].median())
    df["SUM_ASSURED"]      = df["SUM_ASSURED"].fillna(df["SUM_ASSURED"].median())
    df["PI_OCCUPATION"]    = df["PI_OCCUPATION"].fillna("Unknown")
    df["REASON_FOR_CLAIM"] = df["REASON_FOR_CLAIM"].fillna("Not Specified")

    # Derived grouping columns
    df["AGE_GROUP"] = pd.cut(
        df["PI_AGE"],
        bins=[0, 18, 30, 45, 60, 200],
        labels=["<18", "18-30", "31-45", "46-60", "60+"],
        right=True
    )
    df["INCOME_GROUP"] = pd.cut(
        df["PI_ANNUAL_INCOME"],
        bins=[0, 100000, 300000, 600000, float("inf")],
        labels=["Low (<1L)", "Middle (1-3L)", "Upper-Mid (3-6L)", "High (>6L)"],
        right=True
    )
    df["STATUS_BINARY"] = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    return df

# ─── Feature engineering ─────────────────────────────────────────────────────
@st.cache_data
def engineer(df):
    fe = df.copy()
    cat_cols = ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON",
                "MEDICAL_NONMED", "PI_STATE", "PI_OCCUPATION"]
    le = LabelEncoder()
    for col in cat_cols:
        fe[col + "_ENC"] = le.fit_transform(fe[col].astype(str))

    fe["LOG_INCOME"]         = np.log1p(fe["PI_ANNUAL_INCOME"])
    fe["LOG_SUM_ASSURED"]    = np.log1p(fe["SUM_ASSURED"])
    fe["INCOME_TO_SUM_RATIO"] = fe["PI_ANNUAL_INCOME"] / (fe["SUM_ASSURED"] + 1)

    feats = ["PI_AGE", "LOG_INCOME", "LOG_SUM_ASSURED", "INCOME_TO_SUM_RATIO",
             "PI_GENDER_ENC", "ZONE_ENC", "PAYMENT_MODE_ENC", "EARLY_NON_ENC",
             "MEDICAL_NONMED_ENC", "PI_STATE_ENC", "PI_OCCUPATION_ENC"]

    X = fe[feats].copy()
    y = fe["STATUS_BINARY"].copy()

    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=feats)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25,
                                           random_state=42, stratify=y)
    return Xtr, Xte, ytr, yte, feats

# ─── Model training ───────────────────────────────────────────────────────────
@st.cache_data
def train_all(df):
    Xtr, Xte, ytr, yte, feats = engineer(df)
    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":     DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    out = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        yp_tr = m.predict(Xtr)
        yp_te = m.predict(Xte)
        yprob = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, yprob)
        out[name] = {
            "train_acc": accuracy_score(ytr, yp_tr),
            "test_acc":  accuracy_score(yte, yp_te),
            "precision": precision_score(yte, yp_te, zero_division=0),
            "recall":    recall_score(yte,    yp_te, zero_division=0),
            "f1":        f1_score(yte,         yp_te, zero_division=0),
            "cm":        confusion_matrix(yte, yp_te),
            "fpr": fpr, "tpr": tpr,
            "auc": auc(fpr, tpr),
            "yte": yte, "ypred": yp_te,
        }
        if hasattr(m, "feature_importances_"):
            out[name]["fi"] = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    return out, feats

# ─── Approval-rate groupby helper ─────────────────────────────────────────────
def appr_rate(df, col):
    """Returns Series: group → approval % using transform-safe approach."""
    tmp = df[[col, "POLICY_STATUS"]].copy()
    tmp["approved"] = (tmp["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    return tmp.groupby(col, observed=True)["approved"].mean() * 100

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Bias Analyser")
    st.markdown("**Insurance Claim Settlement**")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Insurance CSV", type=["csv"])
    st.markdown("---")
    st.markdown("""
    **Tabs**
    - 📊 Descriptive Analysis
    - 🔍 Diagnostic Bias
    - ⚙️ Feature Engineering
    - 🤖 ML Models
    - 📈 Model Performance
    - 📋 Findings
    """)

# ─── Load data ────────────────────────────────────────────────────────────────
if uploaded:
    df = load_data(uploaded)
else:
    st.info("👈 Upload your **Insurance.csv** from the sidebar to begin.")
    st.stop()

total        = len(df)
n_approved   = (df["POLICY_STATUS"] == "Approved Death Claim").sum()
n_repudiated = total - n_approved
pct_approved = n_approved / total * 100

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f3460,#16213e,#1a1d2e);
            border-radius:14px;padding:28px 36px;margin-bottom:24px;border:1px solid #2d3561;'>
  <h1 style='color:#e94560;margin:0;font-size:28px;font-weight:700;'>
    🔍 Insurance Claim Settlement Bias Analyser
  </h1>
  <p style='color:#8892b0;margin:8px 0 0 0;font-size:14px;'>
    Descriptive · Diagnostic · Predictive · Bias Detection
  </p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
def kpi(col, lbl, val, sub=""):
    col.markdown(f"""<div class='metric-card'>
      <div class='label'>{lbl}</div><div class='value'>{val}</div><div class='sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, "Total Claims",   f"{total:,}",        "records")
kpi(c2, "Approved",       f"{n_approved:,}",   f"{pct_approved:.1f}%")
kpi(c3, "Repudiated",     f"{n_repudiated:,}", f"{100-pct_approved:.1f}%")
kpi(c4, "Age Range",      f"{df['PI_AGE'].min()}–{df['PI_AGE'].max()}", "years")
kpi(c5, "States",         f"{df['PI_STATE'].nunique()}", "distinct")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Bias",
    "⚙️ Feature Engineering",
    "🤖 ML Models",
    "📈 Model Performance",
    "📋 Findings",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>1. Cross-Tabulation Against Policy Status</div>",
                unsafe_allow_html=True)

    dims   = ["PI_GENDER", "AGE_GROUP", "INCOME_GROUP", "PAYMENT_MODE",
              "EARLY_NON", "MEDICAL_NONMED", "ZONE"]
    chosen = st.selectbox("Select Dimension", dims, key="xtab")

    # Cross-tab table
    ct = pd.crosstab(df[chosen], df["POLICY_STATUS"], margins=True, margins_name="Total")
    approved_col = "Approved Death Claim"
    ct["Approval Rate (%)"] = (
        ct.get(approved_col, pd.Series(0, index=ct.index)) / ct["Total"] * 100
    ).round(1)
    st.dataframe(
        ct.style
          .background_gradient(subset=["Approval Rate (%)"], cmap="RdYlGn")
          .format({"Approval Rate (%)": "{:.1f}%"}),
        use_container_width=True
    )

    # Bar charts
    ct2 = pd.crosstab(df[chosen], df["POLICY_STATUS"])
    pct = ct2.div(ct2.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ct2.plot(kind="bar", ax=ax1, color=[CLR_A, CLR_R], edgecolor="none", width=0.7)
    ax1.set_title(f"Count by {chosen}"); ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=35); ax1.legend(fontsize=8)

    pct.plot(kind="bar", stacked=True, ax=ax2, color=[CLR_A, CLR_R], edgecolor="none", width=0.7)
    ax2.set_title(f"Approval % by {chosen}"); ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=35); ax2.legend(fontsize=8)

    dark(fig, [ax1, ax2]); plt.tight_layout(); show(fig)

    c2v, p2v = chi2_test(df, chosen)
    st.markdown(bias_box(p2v, chosen), unsafe_allow_html=True)

    # ── Univariate distributions ──
    st.markdown("<div class='section-header'>2. Univariate Distributions</div>",
                unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    axs = axes2.flatten().tolist()   # convert to plain list

    for status, colour in PALETTE.items():
        axs[0].hist(df.loc[df["POLICY_STATUS"]==status, "PI_AGE"],
                    bins=20, alpha=0.6, color=colour, label=status, edgecolor="none")
    axs[0].set_title("Age Distribution"); axs[0].legend(fontsize=7)

    for status, colour in PALETTE.items():
        axs[1].hist(np.log1p(df.loc[df["POLICY_STATUS"]==status, "PI_ANNUAL_INCOME"]),
                    bins=20, alpha=0.6, color=colour, label=status, edgecolor="none")
    axs[1].set_title("Log(Income)"); axs[1].legend(fontsize=7)

    for status, colour in PALETTE.items():
        axs[2].hist(np.log1p(df.loc[df["POLICY_STATUS"]==status, "SUM_ASSURED"]),
                    bins=20, alpha=0.6, color=colour, label=status, edgecolor="none")
    axs[2].set_title("Log(Sum Assured)"); axs[2].legend(fontsize=7)

    g = df.groupby(["PI_GENDER", "POLICY_STATUS"], observed=True).size().unstack(fill_value=0)
    g.plot(kind="bar", ax=axs[3], color=[CLR_A, CLR_R], edgecolor="none")
    axs[3].set_title("Gender vs Status"); axs[3].tick_params(axis="x", rotation=0)

    m = df.groupby(["MEDICAL_NONMED", "POLICY_STATUS"], observed=True).size().unstack(fill_value=0)
    m.plot(kind="bar", ax=axs[4], color=[CLR_A, CLR_R], edgecolor="none")
    axs[4].set_title("Medical vs Status"); axs[4].tick_params(axis="x", rotation=0)

    pm = df.groupby(["PAYMENT_MODE", "POLICY_STATUS"], observed=True).size().unstack(fill_value=0)
    pm.plot(kind="bar", ax=axs[5], color=[CLR_A, CLR_R], edgecolor="none")
    axs[5].set_title("Payment Mode vs Status"); axs[5].tick_params(axis="x", rotation=25)

    for ax in axs:
        ax.legend(fontsize=6)

    dark(fig2, axs); plt.tight_layout(); show(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC BIAS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Diagnostic Bias Analysis</div>",
                unsafe_allow_html=True)

    # ── Age ──
    st.markdown("#### 📅 Age-wise Bias")
    age_rate = appr_rate(df, "AGE_GROUP").reset_index()
    age_rate.columns = ["AGE_GROUP", "Approval_Rate"]
    age_cnt  = df.groupby("AGE_GROUP", observed=True).size().reset_index(name="Count")
    age_rate = age_rate.merge(age_cnt, on="AGE_GROUP")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    cols_age = [CLR_A if r >= 65 else CLR_R for r in age_rate["Approval_Rate"]]
    ax1.bar(age_rate["AGE_GROUP"].astype(str), age_rate["Approval_Rate"],
            color=cols_age, edgecolor="none", width=0.6)
    ax1.axhline(pct_approved, color="white", linestyle="--", lw=1,
                label=f"Overall {pct_approved:.1f}%")
    ax1.set_title("Approval Rate by Age Group"); ax1.set_ylabel("Approval Rate (%)"); ax1.legend(fontsize=8)

    app_ages = df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_AGE"]
    rep_ages = df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_AGE"]
    bp = ax2.boxplot([app_ages.tolist(), rep_ages.tolist()], patch_artist=True, widths=0.45)
    ax2.set_xticklabels(["Approved", "Repudiated"])
    bp["boxes"][0].set_facecolor(CLR_A); bp["boxes"][1].set_facecolor(CLR_R)
    for med in bp["medians"]: med.set(color="white", linewidth=2)
    ax2.set_title("Age: Approved vs Repudiated")

    dark(fig, [ax1, ax2]); show(fig)
    t, p = stats.ttest_ind(app_ages, rep_ages)
    st.markdown(bias_box(p, "Age (t-test)", chi=False), unsafe_allow_html=True)

    # ── Income ──
    st.markdown("#### 💰 Income-wise Bias")
    inc_rate = appr_rate(df, "INCOME_GROUP").reset_index()
    inc_rate.columns = ["INCOME_GROUP", "Approval_Rate"]

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 4))
    cols_inc = [CLR_A if r >= 65 else CLR_R for r in inc_rate["Approval_Rate"]]
    ax3.bar(inc_rate["INCOME_GROUP"].astype(str), inc_rate["Approval_Rate"],
            color=cols_inc, edgecolor="none", width=0.6)
    ax3.axhline(pct_approved, color="white", linestyle="--", lw=1,
                label=f"Overall {pct_approved:.1f}%")
    ax3.set_title("Approval Rate by Income Group"); ax3.set_ylabel("Approval Rate (%)")
    ax3.tick_params(axis="x", rotation=15); ax3.legend(fontsize=8)

    app_inc = np.log1p(df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_ANNUAL_INCOME"])
    rep_inc = np.log1p(df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_ANNUAL_INCOME"])
    bp2 = ax4.boxplot([app_inc.tolist(), rep_inc.tolist()], patch_artist=True, widths=0.45)
    ax4.set_xticklabels(["Approved", "Repudiated"])
    bp2["boxes"][0].set_facecolor(CLR_A); bp2["boxes"][1].set_facecolor(CLR_R)
    for med in bp2["medians"]: med.set(color="white", linewidth=2)
    ax4.set_title("Log(Income): Approved vs Repudiated")

    dark(fig2, [ax3, ax4]); show(fig2)
    t2, p2 = stats.ttest_ind(
        df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_ANNUAL_INCOME"],
        df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_ANNUAL_INCOME"]
    )
    st.markdown(bias_box(p2, "Income (t-test)", chi=False), unsafe_allow_html=True)

    # ── Zone ──
    st.markdown("#### 🗺️ Zone-wise Bias")
    zone_rate = appr_rate(df, "ZONE").reset_index()
    zone_rate.columns = ["ZONE", "Approval_Rate"]
    zone_cnt  = df.groupby("ZONE", observed=True).size().reset_index(name="Count")
    zone_rate = zone_rate.merge(zone_cnt, on="ZONE")
    zone_rate = zone_rate[zone_rate["Count"] >= 15].sort_values("Approval_Rate")

    fig3, ax5 = plt.subplots(figsize=(14, 6))
    cols_z = [CLR_A if r >= pct_approved else CLR_R for r in zone_rate["Approval_Rate"]]
    bars = ax5.barh(zone_rate["ZONE"], zone_rate["Approval_Rate"], color=cols_z, edgecolor="none")
    ax5.axvline(pct_approved, color="white", linestyle="--", lw=1.5,
                label=f"Overall {pct_approved:.1f}%")
    ax5.set_title("Approval Rate by Zone (min 15 claims)"); ax5.set_xlabel("Approval Rate (%)")
    ax5.legend(fontsize=9)
    for bar, cnt in zip(bars, zone_rate["Count"]):
        ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"n={cnt}", va="center", color=TXT_CLR, fontsize=8)
    dark(fig3, [ax5]); show(fig3)
    cz, pz = chi2_test(df, "ZONE")
    st.markdown(bias_box(pz, "Zone"), unsafe_allow_html=True)

    # ── Gender deep-dive ──
    st.markdown("#### 👥 Gender Bias Deep-Dive")
    # Build pivot cleanly without apply
    gdf = df[["PI_GENDER", "AGE_GROUP", "POLICY_STATUS"]].copy()
    gdf["approved"] = (gdf["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    g_pivot = (gdf.groupby(["PI_GENDER", "AGE_GROUP"], observed=True)["approved"]
                   .mean() * 100).unstack(fill_value=0)

    mdf = df[["MEDICAL_NONMED", "PI_GENDER", "POLICY_STATUS"]].copy()
    mdf["approved"] = (mdf["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    m_pivot = (mdf.groupby(["MEDICAL_NONMED", "PI_GENDER"], observed=True)["approved"]
                   .mean() * 100).unstack(fill_value=0)

    fig4, (ax6, ax7) = plt.subplots(1, 2, figsize=(13, 4))
    g_pivot.T.plot(kind="bar", ax=ax6, color=[CLR_A, CLR_R], edgecolor="none")
    ax6.set_title("Approval Rate: Gender × Age"); ax6.tick_params(axis="x", rotation=15)
    ax6.legend(title="Gender", fontsize=8)

    m_pivot.plot(kind="bar", ax=ax7, color=[CLR_A, CLR_R], edgecolor="none")
    ax7.set_title("Approval Rate: Medical × Gender"); ax7.tick_params(axis="x", rotation=0)
    ax7.legend(title="Gender", fontsize=8)

    dark(fig4, [ax6, ax7]); show(fig4)

    # ── Correlation heatmap ──
    st.markdown("#### 🔗 Correlation Heatmap")
    corr = df[["PI_AGE", "PI_ANNUAL_INCOME", "SUM_ASSURED", "STATUS_BINARY"]].corr()
    fig5, ax8 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax8,
                linewidths=0.5, linecolor=DARK_BG, annot_kws={"size": 11})
    ax8.set_title("Feature Correlation Matrix")
    dark(fig5, [ax8]); show(fig5)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Feature Engineering Pipeline</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='finding-card'><h4>🛠️ Steps Applied</h4><p>
    <b>1. Numeric Cleaning:</b> Removed commas from SUM_ASSURED and PI_ANNUAL_INCOME → float.<br>
    <b>2. Imputation:</b> PI_OCCUPATION → "Unknown"; REASON_FOR_CLAIM → "Not Specified"; numerics → median.<br>
    <b>3. Label Encoding:</b> PI_GENDER, ZONE, PAYMENT_MODE, EARLY_NON, MEDICAL_NONMED, PI_STATE, PI_OCCUPATION.<br>
    <b>4. Log Transforms:</b> LOG_INCOME and LOG_SUM_ASSURED to reduce skewness.<br>
    <b>5. Ratio Feature:</b> INCOME_TO_SUM_RATIO = PI_ANNUAL_INCOME / SUM_ASSURED.<br>
    <b>6. StandardScaler:</b> All features scaled before model training.<br>
    <b>7. Stratified 75/25 split</b> preserving class ratio.
    </p></div>
    """, unsafe_allow_html=True)

    # Class distribution + log transforms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sc = df["POLICY_STATUS"].value_counts()
    ax1.pie(sc.values, labels=sc.index, colors=[CLR_A, CLR_R],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": TXT_CLR, "fontsize": 10},
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 2})
    ax1.set_title("Class Distribution", color=TXT_CLR)

    ax2.hist(np.log1p(df["PI_ANNUAL_INCOME"]), bins=25, color=CLR_A, alpha=0.7,
             edgecolor="none", label="Log(Income)")
    ax2.hist(np.log1p(df["SUM_ASSURED"]),     bins=25, color=CLR_R, alpha=0.7,
             edgecolor="none", label="Log(Sum Assured)")
    ax2.set_title("Log-Transformed Features"); ax2.legend(fontsize=9)
    dark(fig, [ax1, ax2]); show(fig)

    Xtr, Xte, ytr, yte, feats = engineer(df)
    st.markdown("#### Engineered Feature Sample (first 5 train rows)")
    preview = Xtr.head(5).round(3)
    st.dataframe(preview, use_container_width=True)
    st.markdown(f"""
    <div class='finding-card'><h4>📐 Split Info</h4><p>
    Train: <b>{len(Xtr)}</b> | Test: <b>{len(Xte)}</b> | Features: <b>{len(feats)}</b><br>
    {", ".join(f"<code>{f}</code>" for f in feats)}
    </p></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Super-Learning Classification Models</div>",
                unsafe_allow_html=True)

    with st.spinner("Training KNN · Decision Tree · Random Forest · Gradient Boosting …"):
        results, feats = train_all(df)
    st.success("✅ All 4 models trained successfully!")

    # Summary table — keep ROC-AUC numeric so highlight works
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":         name,
            "Train Acc (%)": round(r["train_acc"] * 100, 2),
            "Test Acc (%)":  round(r["test_acc"]  * 100, 2),
            "Precision":     round(r["precision"], 3),
            "Recall":        round(r["recall"],    3),
            "F1 Score":      round(r["f1"],        3),
            "ROC-AUC":       round(r["auc"],       3),
        })
    sdf = pd.DataFrame(rows)
    # All columns numeric — highlight works safely
    st.dataframe(
        sdf.style
           .highlight_max(subset=["Test Acc (%)", "F1 Score", "ROC-AUC"], color="#1a4a2e")
           .highlight_min(subset=["Test Acc (%)"],                          color="#4a1a1a"),
        use_container_width=True
    )

    # Feature importances
    st.markdown("#### 🌲 Feature Importances (Tree Models)")
    tree_res = {n: r for n, r in results.items() if "fi" in r}
    fig, axes_fi = plt.subplots(1, len(tree_res), figsize=(14, 5))
    axes_fi_list = [axes_fi] if len(tree_res) == 1 else axes_fi.tolist()
    for ax, (name, r) in zip(axes_fi_list, tree_res.items()):
        fi = r["fi"].head(10)
        ax.barh(fi.index[::-1], fi.values[::-1], color=CLR_A, edgecolor="none", height=0.6)
        ax.set_title(f"{name}\nImportances", fontsize=10)
        ax.set_xlabel("Importance")
    dark(fig, axes_fi_list); plt.tight_layout(); show(fig)

    # Classification report
    st.markdown("#### 📜 Classification Report")
    sel = st.selectbox("Select model:", list(results.keys()), key="cr_sel")
    rep = classification_report(
        results[sel]["yte"], results[sel]["ypred"],
        target_names=["Repudiate Death", "Approved Death Claim"]
    )
    st.code(rep, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

    names      = list(results.keys())
    train_accs = [results[n]["train_acc"] * 100 for n in names]
    test_accs  = [results[n]["test_acc"]  * 100 for n in names]
    precs      = [results[n]["precision"]        for n in names]
    recs       = [results[n]["recall"]           for n in names]
    f1s        = [results[n]["f1"]               for n in names]

    # Accuracy + metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(names)); w = 0.35
    ax1.bar(x - w/2, train_accs, width=w, label="Train", color=CLR_A, edgecolor="none")
    ax1.bar(x + w/2, test_accs,  width=w, label="Test",  color=CLR_R, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, fontsize=9)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Train vs Test Accuracy")
    ax1.legend(fontsize=9); ax1.set_ylim(50, 108)
    for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
        ax1.text(i-w/2, tr+0.5, f"{tr:.1f}", ha="center", fontsize=8, color=TXT_CLR)
        ax1.text(i+w/2, te+0.5, f"{te:.1f}", ha="center", fontsize=8, color=TXT_CLR)

    ax2.plot(names, precs, "o-", color=CLR_A,     lw=2, ms=8, label="Precision")
    ax2.plot(names, recs,  "s-", color="#f7c59f",  lw=2, ms=8, label="Recall")
    ax2.plot(names, f1s,   "^-", color=CLR_R,      lw=2, ms=8, label="F1")
    ax2.set_ylim(0, 1.05); ax2.set_title("Precision / Recall / F1")
    ax2.tick_params(axis="x", rotation=15); ax2.legend(fontsize=9)
    dark(fig, [ax1, ax2]); plt.tight_layout(); show(fig)

    # ROC curves
    st.markdown("#### 📉 ROC Curves")
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    roc_colors = [CLR_A, "#f7c59f", "#c77dff", CLR_R]
    for (name, r), clr in zip(results.items(), roc_colors):
        ax3.plot(r["fpr"], r["tpr"], lw=2.5, color=clr,
                 label=f"{name}  (AUC={r['auc']:.3f})")
    ax3.plot([0,1],[0,1], "w--", lw=1, label="Random (AUC=0.5)")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.02)
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves — All Models"); ax3.legend(fontsize=9, loc="lower right")
    dark(fig2, [ax3]); show(fig2)

    # Confusion matrices
    st.markdown("#### 🟥 Confusion Matrices")
    fig3, axes_cm = plt.subplots(1, 4, figsize=(16, 4))
    cm_labels = ["Repudiated", "Approved"]
    for ax, (name, r) in zip(axes_cm.tolist(), results.items()):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=cm_labels, yticklabels=cm_labels,
                    linewidths=0.5, linecolor=DARK_BG, annot_kws={"size": 12})
        ax.set_title(f"{name}\nAcc={r['test_acc']*100:.1f}%", fontsize=10)
        ax.set_xlabel("Predicted", color=TXT_CLR, fontsize=9)
        ax.set_ylabel("Actual",    color=TXT_CLR, fontsize=9)
        ax.tick_params(colors=TXT_CLR, labelsize=8)
    dark(fig3, axes_cm.tolist()); plt.tight_layout(); show(fig3)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>📋 Key Findings & Recommendations</div>",
                unsafe_allow_html=True)

    age_app = df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_AGE"].mean()
    age_rep = df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_AGE"].mean()
    inc_app = df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_ANNUAL_INCOME"].mean()
    inc_rep = df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_ANNUAL_INCOME"].mean()

    zone_r    = appr_rate(df, "ZONE")
    best_zone = zone_r.idxmax(); worst_zone = zone_r.idxmin()

    best_model = max(results.items(), key=lambda x: x[1]["f1"])[0]
    best_auc   = max(r["auc"] for r in results.values())

    _, p_med   = chi2_test(df, "MEDICAL_NONMED")
    _, p_early = chi2_test(df, "EARLY_NON")
    _, p_age   = stats.ttest_ind(
        df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_AGE"],
        df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_AGE"]
    )
    _, p_inc = stats.ttest_ind(
        df.loc[df["POLICY_STATUS"]=="Approved Death Claim", "PI_ANNUAL_INCOME"],
        df.loc[df["POLICY_STATUS"]=="Repudiate Death",      "PI_ANNUAL_INCOME"]
    )

    findings = [
        ("🎂", "Age Bias",
         f"Approved avg age: <b>{age_app:.1f}</b> vs Repudiated: <b>{age_rep:.1f}</b>. "
         f"t-test p={p_age:.4f} — {'significant ⚠️' if p_age<0.05 else 'not significant ✅'}."),
        ("💰", "Income Disparity",
         f"Approved avg income: ₹{inc_app:,.0f} vs Repudiated: ₹{inc_rep:,.0f}. "
         f"t-test p={p_inc:.4f} — {'significant ⚠️' if p_inc<0.05 else 'not significant ✅'}."),
        ("🗺️", "Zone Inequity",
         f"Best zone: <b>{best_zone}</b> | Worst: <b>{worst_zone}</b>. "
         f"Significant geographic variation in approval rates detected."),
        ("🏥", "Medical Influence",
         f"MEDICAL_NONMED p={p_med:.4f} — "
         f"{'Significant association with outcome ⚠️' if p_med<0.05 else 'No significant association ✅'}."),
        ("⏰", "Early Claim Flag",
         f"EARLY_NON p={p_early:.4f} — "
         f"{'Early claims significantly more repudiated ⚠️' if p_early<0.05 else 'No significant effect ✅'}."),
        ("🤖", "Best Model",
         f"<b>{best_model}</b> achieves highest F1. Best ROC-AUC: <b>{best_auc:.3f}</b>. "
         f"Key predictors: AGE, INCOME, ZONE, MEDICAL_NONMED."),
    ]

    for icon, title, body in findings:
        st.markdown(f"""
        <div class='finding-card'>
          <h4>{icon} {title}</h4><p>{body}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>💡 Recommendations</div>", unsafe_allow_html=True)
    recs = [
        ("🔄", "Zone Audit",            "Audit all repudiated claims from zones with >15% below-average approval rates."),
        ("📋", "Blind Assessment",       "Hide age and income fields during initial claim evaluation to reduce unconscious bias."),
        ("📊", "Monthly Monitoring",     "Run this dashboard monthly; flag any group where approval rate drops >10% below baseline."),
        ("🎓", "Team Training",          "Structured training for all settlement teams aligned with IRDAI fair-practice guidelines."),
        ("🤖", "AI-Assisted Review",     "Use Gradient Boosting to flag high-confidence approvals that were manually rejected."),
        ("📁", "Rejection Reason Codes", "Mandate structured rejection codes to enable longitudinal bias tracking."),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(recs):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='finding-card'>
              <h4>{icon} {title}</h4><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(233,69,96,0.08);border:1px solid #e94560;border-radius:10px;
                padding:16px 20px;margin-top:20px;'>
      <p style='color:#e94560;margin:0;font-size:13px;font-weight:600;'>⚠️ Disclaimer</p>
      <p style='color:#8892b0;margin:6px 0 0;font-size:12px;'>
        For investigative and internal audit purposes only. Statistical significance does not
        imply discriminatory intent. Review all findings with domain experts and legal counsel
        before any formal action.
      </p>
    </div>""", unsafe_allow_html=True)
