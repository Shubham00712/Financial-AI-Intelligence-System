# import json
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st
# from pathlib import Path
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# st.set_page_config(page_title="Bank Fraud & KYC Dashboard", page_icon="🛡️", layout="wide")

# OUT = Path("output")
# MODEL_PATH = OUT / "fraud_model.joblib"
# METRICS_PATH = OUT / "metrics.json"
# FEATURES_PATH = OUT / "feature_columns.json"
# CAT_COLS_PATH = OUT / "cat_columns.json"
# MEDIANS_PATH = OUT / "num_medians.json"
# DATA_PATH = Path("synthetic_bank_fraud_30k.csv")
# SCORDED_PATH = OUT / "scored_data.csv"

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #07111f 0%, #0b1b2e 45%, #10263f 100%);
#         color: #edf2f7;
#     }
#     section[data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #08111f 0%, #0b1726 100%);
#         border-right: 1px solid rgba(255,255,255,0.08);
#     }
#     .main .block-container {
#         padding-top: 1.2rem;
#         padding-bottom: 2rem;
#     }
#     [data-testid="stHeader"] {
#         background: rgba(0,0,0,0);
#     }
#     div[data-testid="metric-container"] {
#         background: rgba(10, 22, 38, 0.88);
#         border: 1px solid rgba(255,255,255,0.08);
#         border-radius: 16px;
#         padding: 12px 14px;
#         box-shadow: 0 12px 28px rgba(0,0,0,0.22);
#     }
#     .stButton button {
#         background: linear-gradient(135deg, #22c55e, #16a34a);
#         color: white;
#         border: 0;
#         border-radius: 12px;
#         padding: 0.55rem 1rem;
#         font-weight: 600;
#     }
#     .stButton button:hover {
#         background: linear-gradient(135deg, #34d399, #15803d);
#     }
#     [data-testid="stForm"] {
#         background: rgba(8, 17, 31, 0.88);
#         border: 1px solid rgba(255,255,255,0.08);
#         border-radius: 18px;
#         padding: 18px;
#     }
#     [data-testid="stDataFrame"] {
#         background: rgba(255,255,255,0.98);
#         border-radius: 12px;
#     }
#     h1, h2, h3, h4, p, label, span {
#         color: #edf2f7;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# @st.cache_resource
# def load_artifacts():
#     model = joblib.load(MODEL_PATH)
#     metrics = json.load(open(METRICS_PATH))
#     feature_cols = json.load(open(FEATURES_PATH))
#     cat_cols = json.load(open(CAT_COLS_PATH))
#     medians = json.load(open(MEDIANS_PATH))
#     df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.read_csv(SCORDED_PATH)
#     return model, metrics, feature_cols, cat_cols, medians, df


# def make_model_input(row_df, feature_cols, cat_cols, medians):
#     row = row_df.copy()
#     if "customer_id" in row.columns:
#         row = row.drop(columns=["customer_id"])
#     for c, v in medians.items():
#         if c in row.columns:
#             row[c] = row[c].fillna(v)
#     for c in cat_cols:
#         if c in row.columns:
#             row[c] = row[c].fillna("Unknown")
#     row = pd.get_dummies(row, columns=[c for c in cat_cols if c in row.columns], drop_first=False)
#     return row.reindex(columns=feature_cols, fill_value=0)

# model, metrics, feature_cols, cat_cols, medians, df = load_artifacts()

# if "fraud_probability" not in df.columns:
#     base = make_model_input(df.drop(columns=["fraud_label"], errors="ignore"), feature_cols, cat_cols, medians)
#     df["fraud_probability"] = model.predict_proba(base)[:, 1]

# df["kyc_status"] = np.where(df["kyc_verified"].astype(int) == 1, "Verified", "Not Verified")
# df["fraud_flag"] = np.where(df["fraud_probability"] >= 0.5, "High Risk", "Low Risk")

# st.title("🛡️ Bank Fraud Detection & KYC Dashboard")
# st.caption("AI/ML fraud scoring, KYC verification risk, customer analytics, and case review dashboard.")

# with st.sidebar:
#     st.header("Filters")
#     age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
#     kyc_filter = st.multiselect("KYC Status", sorted(df["kyc_status"].unique().tolist()), default=sorted(df["kyc_status"].unique().tolist()))
#     risk_min, risk_max = st.slider("Fraud Probability", 0.0, 1.0, (0.0, 1.0), 0.01)
#     selected_region = st.multiselect("Region", sorted(df["region"].unique().tolist()), default=sorted(df["region"].unique().tolist()))

# filtered = df[
#     (df["age"].between(age_range[0], age_range[1])) &
#     (df["kyc_status"].isin(kyc_filter)) &
#     (df["fraud_probability"].between(risk_min, risk_max)) &
#     (df["region"].isin(selected_region))
# ].copy()

# k1, k2, k3, k4, k5 = st.columns(5)
# k1.metric("Customers", f"{len(df):,}")
# k2.metric("Fraud Rate", f"{df['fraud_label'].mean()*100:.2f}%")
# k3.metric("KYC Pass Rate", f"{df['kyc_verified'].mean()*100:.2f}%")
# k4.metric("Avg Fraud Risk", f"{df['fraud_probability'].mean()*100:.2f}%")
# k5.metric("High Risk Cases", f"{int((df['fraud_probability']>=0.5).sum()):,}")

# m1, m2, m3, m4 = st.columns(4)
# m1.metric("Model", metrics["best_model"])
# m2.metric("Accuracy", f"{metrics['selected_metrics']['accuracy']:.3f}")
# m3.metric("ROC AUC", f"{metrics['selected_metrics']['roc_auc']:.3f}")
# m4.metric("F1", f"{metrics['selected_metrics']['f1']:.3f}")

# left, right = st.columns([1.2, 0.8])
# with left:
#     st.subheader("Customer Distribution")
#     fig = make_subplots(rows=2, cols=2, subplot_titles=("Credit Score", "Annual Income", "Debt-to-Income", "Fraud Probability"))
#     fig.add_trace(go.Histogram(x=filtered["credit_score"], nbinsx=35, marker_color="#38bdf8", opacity=0.88), row=1, col=1)
#     fig.add_trace(go.Histogram(x=filtered["annual_income"], nbinsx=35, marker_color="#a78bfa", opacity=0.88), row=1, col=2)
#     fig.add_trace(go.Histogram(x=filtered["debt_to_income_ratio"], nbinsx=35, marker_color="#f59e0b", opacity=0.88), row=2, col=1)
#     fig.add_trace(go.Histogram(x=filtered["fraud_probability"], nbinsx=35, marker_color="#ef4444", opacity=0.88), row=2, col=2)
#     fig.update_layout(
#         height=700,
#         showlegend=False,
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(255,255,255,0.03)",
#         font=dict(color="#edf2f7"),
#         margin=dict(l=20, r=20, t=60, b=20),
#     )
#     st.plotly_chart(fig, use_container_width=True)
# with right:
#     st.subheader("Model Metrics")
#     st.json(metrics)
#     st.subheader("Fraud vs KYC")
#     pie = pd.DataFrame({"label": ["Verified", "Not Verified"], "value": [int((df["kyc_verified"] == 1).sum()), int((df["kyc_verified"] == 0).sum())]})
#     st.plotly_chart(px.pie(pie, names="label", values="value", hole=0.58, color_discrete_sequence=["#22c55e", "#ef4444"]), use_container_width=True)
#     region_perf = df.groupby("region", as_index=False)["fraud_probability"].mean().sort_values("fraud_probability", ascending=False)
#     st.plotly_chart(
#         px.bar(
#             region_perf,
#             x="region",
#             y="fraud_probability",
#             title="Avg Fraud Probability by Region",
#             color="fraud_probability",
#             color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
#         ),
#         use_container_width=True,
#     )

# st.subheader("Behavioral Relationships")
# colx, coly = st.columns(2)
# with colx:
#     st.plotly_chart(
#         px.scatter(
#             filtered,
#             x="employment_years",
#             y="fraud_probability",
#             color="fraud_flag",
#             size="loan_amount",
#             hover_data=["age", "annual_income", "credit_score", "region"],
#             color_discrete_map={"High Risk": "#ef4444", "Low Risk": "#22c55e"},
#             title="Employment Years vs Fraud Probability",
#         ),
#         use_container_width=True,
#     )
# with coly:
#     st.plotly_chart(
#         px.scatter(
#             filtered,
#             x="credit_score",
#             y="debt_to_income_ratio",
#             color="fraud_flag",
#             size="fraud_probability",
#             hover_data=["kyc_status", "region"],
#             color_discrete_map={"High Risk": "#ef4444", "Low Risk": "#22c55e"},
#             title="Credit Score vs DTI",
#         ),
#         use_container_width=True,
#     )

# st.subheader("KYC + Fraud Risk Scoring")
# with st.form("risk_form"):
#     a1, a2, a3 = st.columns(3)
#     with a1:
#         age = st.number_input("Age", 18, 90, 35)
#         gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#         marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
#         education_level = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "PhD"])
#         employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
#         region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
#     with a2:
#         customer_tenure_years = st.number_input("Customer Tenure (Years)", 0.0, 35.0, 3.0)
#         annual_income = st.number_input("Annual Income", 0.0, 10000000.0, 50000.0)
#         employment_years = st.number_input("Employment Years", 0.0, 50.0, 5.0)
#         savings_balance = st.number_input("Savings Balance", 0.0, 10000000.0, 10000.0)
#         monthly_expenses = st.number_input("Monthly Expenses", 0.0, 1000000.0, 15000.0)
#         existing_loans = st.number_input("Existing Loans", 0, 10, 1)
#     with a3:
#         num_delinquencies = st.number_input("Num Delinquencies", 0, 20, 0)
#         recent_credit_inquiries = st.number_input("Recent Credit Inquiries", 0, 20, 1)
#         loan_amount = st.number_input("Loan Amount", 0.0, 10000000.0, 25000.0)
#         credit_score = st.number_input("Credit Score", 300, 900, 650)
#         debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.25)
#         kyc_verified = st.selectbox("KYC Verified", [0, 1])
#         submit = st.form_submit_button("Predict Risk")

# if submit:
#     row = pd.DataFrame([{
#         "age": age,
#         "gender": gender,
#         "marital_status": marital_status,
#         "education_level": education_level,
#         "employment_status": employment_status,
#         "region": region,
#         "customer_tenure_years": customer_tenure_years,
#         "annual_income": annual_income,
#         "employment_years": employment_years,
#         "savings_balance": savings_balance,
#         "monthly_expenses": monthly_expenses,
#         "existing_loans": existing_loans,
#         "num_delinquencies": num_delinquencies,
#         "recent_credit_inquiries": recent_credit_inquiries,
#         "loan_amount": loan_amount,
#         "credit_score": credit_score,
#         "debt_to_income_ratio": debt_to_income_ratio,
#         "kyc_verified": kyc_verified,
#         "document_match_score": 0.96 if kyc_verified else 0.55,
#         "face_match_score": 0.95 if kyc_verified else 0.50,
#         "liveness_score": 0.94 if kyc_verified else 0.47,
#         "device_trust_score": 0.88 if kyc_verified else 0.45,
#         "transaction_velocity": 3.0 if kyc_verified else 9.0,
#         "avg_txn_amount": loan_amount * 0.15,
#         "cash_out_ratio": 0.2 if kyc_verified else 0.7,
#         "account_age_months": customer_tenure_years * 12,
#         "geo_risk_score": 0.18 if region in ["North", "West"] else 0.30,
#         "rule_engine_hits": int((num_delinquencies > 0) + (recent_credit_inquiries >= 4) + (debt_to_income_ratio > 0.42)),
#     }])
#     pred_x = make_model_input(row, feature_cols, cat_cols, medians)
#     risk = float(model.predict_proba(pred_x)[0, 1])
#     st.metric("Fraud Probability", f"{risk:.2%}")
#     if kyc_verified == 0:
#         st.error("KYC Not Verified")
#     else:
#         st.success("KYC Verified")
#     if risk >= 0.5:
#         st.error("High Fraud Risk")
#     elif risk >= 0.25:
#         st.warning("Medium Fraud Risk")
#     else:
#         st.success("Low Fraud Risk")

# st.subheader("High Risk Cases")
# show_cols = [c for c in ["customer_id", "age", "region", "annual_income", "credit_score", "loan_amount", "debt_to_income_ratio", "kyc_status", "fraud_probability"] if c in filtered.columns]
# st.dataframe(
#     filtered.loc[filtered["fraud_probability"] >= 0.5, show_cols].sort_values("fraud_probability", ascending=False).head(100),
#     use_container_width=True,
# )
# st.download_button("Download Filtered Data", filtered.to_csv(index=False).encode("utf-8"), "filtered_bank_fraud_data.csv", "text/csv")

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# st.set_page_config(page_title="Bank Fraud & KYC Dashboard", page_icon="🛡️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
FRAUD_DIR = BASE_DIR / "Fraud_Detection"
OUT = FRAUD_DIR / "output"
MODEL_PATH = OUT / "fraud_model.joblib"
METRICS_PATH = OUT / "metrics.json"
FEATURES_PATH = OUT / "feature_columns.json"
CAT_COLS_PATH = OUT / "cat_columns.json"
MEDIANS_PATH = OUT / "num_medians.json"
DATA_PATH = Path("synthetic_bank_fraud_30k.csv")
SCORDED_PATH = OUT / "scored_data.csv"

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #06111d 0%, #0a1930 45%, #102b45 100%);
        color: #edf2f7;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101b 0%, #0b1829 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1.8rem;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    div[data-testid="metric-container"] {
        background: rgba(10, 22, 38, 0.9);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
    }
    .stButton button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #34d399, #15803d);
    }
    [data-testid="stForm"] {
        background: rgba(8, 17, 31, 0.90);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
    }
    [data-testid="stDataFrame"] {
        background: rgba(245,247,250,0.98);
        border-radius: 12px;
    }
    h1, h2, h3, h4, p, label, span {
        color: #edf2f7;
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.80));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 14px 30px rgba(0,0,0,0.28);
        height: 100%;
    }
    .metric-title {
        font-size: 0.86rem;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.15;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    .section-box {
        background: rgba(8, 17, 31, 0.84);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metrics = json.load(open(METRICS_PATH))
    feature_cols = json.load(open(FEATURES_PATH))
    cat_cols = json.load(open(CAT_COLS_PATH))
    medians = json.load(open(MEDIANS_PATH))
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.read_csv(SCORDED_PATH)
    return model, metrics, feature_cols, cat_cols, medians, df


def make_model_input(row_df, feature_cols, cat_cols, medians):
    row = row_df.copy()
    if "customer_id" in row.columns:
        row = row.drop(columns=["customer_id"])
    for c, v in medians.items():
        if c in row.columns:
            row[c] = row[c].fillna(v)
    for c in cat_cols:
        if c in row.columns:
            row[c] = row[c].fillna("Unknown")
    row = pd.get_dummies(row, columns=[c for c in cat_cols if c in row.columns], drop_first=False)
    return row.reindex(columns=feature_cols, fill_value=0)

model, metrics, feature_cols, cat_cols, medians, df = load_artifacts()

if "fraud_probability" not in df.columns:
    base = make_model_input(df.drop(columns=["fraud_label"], errors="ignore"), feature_cols, cat_cols, medians)
    df["fraud_probability"] = model.predict_proba(base)[:, 1]

df["kyc_status"] = np.where(df["kyc_verified"].astype(int) == 1, "Verified", "Not Verified")
df["fraud_flag"] = np.where(df["fraud_probability"] >= 0.5, "High Risk", "Low Risk")

st.title("🛡️ Bank Fraud Detection & KYC Dashboard")
st.caption("AI/ML fraud scoring, KYC verification risk, customer analytics, and case review dashboard.")

with st.sidebar:
    st.header("Filters")
    age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    kyc_filter = st.multiselect("KYC Status", sorted(df["kyc_status"].unique().tolist()), default=sorted(df["kyc_status"].unique().tolist()))
    risk_min, risk_max = st.slider("Fraud Probability", 0.0, 1.0, (0.0, 1.0), 0.01)
    selected_region = st.multiselect("Region", sorted(df["region"].unique().tolist()), default=sorted(df["region"].unique().tolist()))

filtered = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["kyc_status"].isin(kyc_filter)) &
    (df["fraud_probability"].between(risk_min, risk_max)) &
    (df["region"].isin(selected_region))
].copy()

st.markdown("### Key Model Metrics")
cm1, cm2, cm3, cm4 = st.columns(4)
cm1.markdown(f"<div class='metric-card'><div class='metric-title'>Model</div><div class='metric-value'>{metrics['best_model']}</div><div class='metric-sub'>selected for deployment</div></div>", unsafe_allow_html=True)
cm2.markdown(f"<div class='metric-card'><div class='metric-title'>Accuracy</div><div class='metric-value'>{metrics['selected_metrics']['accuracy']:.3f}</div><div class='metric-sub'>classification accuracy</div></div>", unsafe_allow_html=True)
cm3.markdown(f"<div class='metric-card'><div class='metric-title'>ROC AUC</div><div class='metric-value'>{metrics['selected_metrics']['roc_auc']:.3f}</div><div class='metric-sub'>ranking quality</div></div>", unsafe_allow_html=True)
cm4.markdown(f"<div class='metric-card'><div class='metric-title'>F1 Score</div><div class='metric-value'>{metrics['selected_metrics']['f1']:.3f}</div><div class='metric-sub'>balance of precision and recall</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Fraud Rate", f"{df['fraud_label'].mean()*100:.2f}%")
k3.metric("KYC Pass Rate", f"{df['kyc_verified'].mean()*100:.2f}%")
k4.metric("Avg Fraud Risk", f"{df['fraud_probability'].mean()*100:.2f}%")
k5.metric("High Risk Cases", f"{int((df['fraud_probability']>=0.5).sum()):,}")

st.markdown("### Customer Distribution")
left, right = st.columns([1.25, 0.75])
with left:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Credit Score", "Annual Income", "Debt-to-Income", "Fraud Probability"),
        vertical_spacing=0.14, horizontal_spacing=0.10
    )
    fig.add_trace(go.Histogram(x=filtered["credit_score"], nbinsx=35, marker_color="#38bdf8", opacity=0.88, showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=filtered["annual_income"], nbinsx=35, marker_color="#a78bfa", opacity=0.88, showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=filtered["debt_to_income_ratio"], nbinsx=35, marker_color="#f59e0b", opacity=0.88, showlegend=False), row=2, col=1)
    fig.add_trace(go.Histogram(x=filtered["fraud_probability"], nbinsx=35, marker_color="#ef4444", opacity=0.88, showlegend=False), row=2, col=2)
    fig.update_layout(
        height=680,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#edf2f7"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("Model Scorecard")
    score_df = pd.DataFrame([
        {"Metric": "Accuracy", "Value": metrics["selected_metrics"]["accuracy"]},
        {"Metric": "Precision", "Value": metrics["selected_metrics"]["precision"]},
        {"Metric": "Recall", "Value": metrics["selected_metrics"]["recall"]},
        {"Metric": "F1 Score", "Value": metrics["selected_metrics"]["f1"]},
        {"Metric": "ROC AUC", "Value": metrics["selected_metrics"]["roc_auc"]},
    ])
    score_fig = px.bar(
        score_df,
        x="Value",
        y="Metric",
        orientation="h",
        text=score_df["Value"].map(lambda x: f"{x:.3f}"),
        color="Value",
        color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
        range_x=[0, 1],
    )
    score_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        showlegend=False,
        font=dict(color="#edf2f7"),
    )
    score_fig.update_traces(textposition="outside")
    st.plotly_chart(score_fig, use_container_width=True)

    st.subheader("KYC Split")
    pie = pd.DataFrame({"label": ["Verified", "Not Verified"], "value": [int((df["kyc_verified"]==1).sum()), int((df["kyc_verified"]==0).sum())]})
    pie_fig = px.pie(pie, names="label", values="value", hole=0.62, color_discrete_sequence=["#22c55e", "#ef4444"])
    pie_fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf2f7"), showlegend=True)
    st.plotly_chart(pie_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Regional & Behavioral Analysis")
colx, coly = st.columns(2)
with colx:
    region_perf = df.groupby("region", as_index=False)["fraud_probability"].mean().sort_values("fraud_probability", ascending=False)
    fig = px.bar(region_perf, x="region", y="fraud_probability", title="Avg Fraud Probability by Region", color="fraud_probability", color_continuous_scale=["#22c55e", "#facc15", "#ef4444"])
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with coly:
    fig = px.scatter(filtered, x="employment_years", y="fraud_probability", color="fraud_flag", size="loan_amount", hover_data=["age", "annual_income", "credit_score", "region"], color_discrete_map={"High Risk": "#ef4444", "Low Risk": "#22c55e"}, title="Employment Years vs Fraud Probability")
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### KYC + Fraud Risk Scoring")
with st.form("risk_form"):
    a1, a2, a3 = st.columns(3)
    with a1:
        age = st.number_input("Age", 18, 90, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education_level = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "PhD"])
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    with a2:
        customer_tenure_years = st.number_input("Customer Tenure (Years)", 0.0, 35.0, 3.0)
        annual_income = st.number_input("Annual Income", 0.0, 10000000.0, 50000.0)
        employment_years = st.number_input("Employment Years", 0.0, 50.0, 5.0)
        savings_balance = st.number_input("Savings Balance", 0.0, 10000000.0, 10000.0)
        monthly_expenses = st.number_input("Monthly Expenses", 0.0, 1000000.0, 15000.0)
        existing_loans = st.number_input("Existing Loans", 0, 10, 1)
    with a3:
        num_delinquencies = st.number_input("Num Delinquencies", 0, 20, 0)
        recent_credit_inquiries = st.number_input("Recent Credit Inquiries", 0, 20, 1)
        loan_amount = st.number_input("Loan Amount", 0.0, 10000000.0, 25000.0)
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.25)
        kyc_verified = st.selectbox("KYC Verified", [0, 1])
        submit = st.form_submit_button("Predict Risk")

if submit:
    row = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education_level": education_level,
        "employment_status": employment_status,
        "region": region,
        "customer_tenure_years": customer_tenure_years,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "savings_balance": savings_balance,
        "monthly_expenses": monthly_expenses,
        "existing_loans": existing_loans,
        "num_delinquencies": num_delinquencies,
        "recent_credit_inquiries": recent_credit_inquiries,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "kyc_verified": kyc_verified,
        "document_match_score": 0.96 if kyc_verified else 0.55,
        "face_match_score": 0.95 if kyc_verified else 0.50,
        "liveness_score": 0.94 if kyc_verified else 0.47,
        "device_trust_score": 0.88 if kyc_verified else 0.45,
        "transaction_velocity": 3.0 if kyc_verified else 9.0,
        "avg_txn_amount": loan_amount * 0.15,
        "cash_out_ratio": 0.2 if kyc_verified else 0.7,
        "account_age_months": customer_tenure_years * 12,
        "geo_risk_score": 0.18 if region in ["North", "West"] else 0.30,
        "rule_engine_hits": int((num_delinquencies > 0) + (recent_credit_inquiries >= 4) + (debt_to_income_ratio > 0.42)),
    }])
    pred_x = make_model_input(row, feature_cols, cat_cols, medians)
    risk = float(model.predict_proba(pred_x)[0, 1])
    st.metric("Fraud Probability", f"{risk:.2%}")
    if kyc_verified == 0:
        st.error("KYC Not Verified")
    else:
        st.success("KYC Verified")
    if risk >= 0.5:
        st.error("High Fraud Risk")
    elif risk >= 0.25:
        st.warning("Medium Fraud Risk")
    else:
        st.success("Low Fraud Risk")

st.markdown("### High Risk Cases")
show_cols = [c for c in ["customer_id", "age", "region", "annual_income", "credit_score", "loan_amount", "debt_to_income_ratio", "kyc_status", "fraud_probability"] if c in filtered.columns]
st.dataframe(
    filtered.loc[filtered["fraud_probability"] >= 0.5, show_cols].sort_values("fraud_probability", ascending=False).head(100),
    use_container_width=True,
)
st.download_button("Download Filtered Data", filtered.to_csv(index=False).encode("utf-8"), "filtered_bank_fraud_data.csv", "text/csv")