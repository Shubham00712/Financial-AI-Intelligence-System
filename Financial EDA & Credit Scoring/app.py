# """
# Financial EDA + Credit Scoring Streamlit app.
# Logic:
# - Load trained model artifacts and synthetic datasets.
# - Provide KYC workflow, customer onboarding, credit scoring, and EDA dashboards.
# - Handle missing files gracefully so the app can be started even before training.
# """

# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# import joblib

# BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# OUT = BASE_DIR / "output"
# DATA_PATH = OUT / "synthetic_banking_data.csv"
# SCORED_PATH = OUT / "synthetic_banking_scored.csv"
# KYC_PATH = OUT / "synthetic_kyc_data.csv"
# MODEL_PATH = OUT / "credit_scoring_model.joblib"
# METRICS_PATH = OUT / "credit_model_metrics.json"

# st.set_page_config(page_title="Financial EDA & Credit Scoring", page_icon="🏦", layout="wide")

# st.markdown("""
# <style>
# .stApp {background: linear-gradient(135deg, #07111c 0%, #0d1f33 45%, #12324d 100%); color: #edf2f7;}
# section[data-testid="stSidebar"] {background: linear-gradient(180deg, #07101b 0%, #0b1829 100%); border-right: 1px solid rgba(255,255,255,0.08);}
# [data-testid="metric-container"] {background: rgba(10, 22, 38, 0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 12px 14px;}
# .stButton button {background: linear-gradient(135deg, #22c55e, #16a34a); color: white; border: 0; border-radius: 12px;}
# h1, h2, h3, h4, p, label, span {color: #edf2f7;}
# .section-box {background: rgba(8, 17, 31, 0.84); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px;}
# </style>
# """, unsafe_allow_html=True)


# def load_assets():
#     data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
#     scored = pd.read_csv(SCORED_PATH) if SCORED_PATH.exists() else data.copy()
#     kyc = pd.read_csv(KYC_PATH) if KYC_PATH.exists() else pd.DataFrame()
#     model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
#     metrics = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else []
#     return data, scored, kyc, model, metrics


# def build_input_frame(form_values):
#     return pd.DataFrame([form_values])


# def kyc_assess(doc_type, doc_status, address_status, selfie_score, sanction_flag, pep_flag, completion):
#     risk = 0
#     risk += 1.3 if doc_status != "Verified" else 0
#     risk += 1.1 if address_status != "Verified" else 0
#     risk += 1.8 if sanction_flag else 0
#     risk += 1.3 if pep_flag else 0
#     risk += 0.8 if selfie_score < 0.75 else 0
#     risk += 0.7 if completion != "Complete" else 0
#     if risk >= 3.2:
#         status = "Rejected"
#         tier = "High"
#     elif risk >= 1.7:
#         status = "Review"
#         tier = "Medium"
#     else:
#         status = "Approved"
#         tier = "Low"
#     return tier, status, round(min(risk / 5.0, 1.0), 3)


# def main():
#     data, scored, kyc, model, metrics = load_assets()
#     st.title("Financial EDA & Credit Scoring")
#     st.caption("Banking-style app for customer onboarding, KYC, analytics, and credit risk scoring.")

#     menu = st.sidebar.radio("Modules", ["Dashboard", "KYC Verification", "Credit Scoring", "Customer Explorer", "Model Metrics"])

#     if menu == "Dashboard":
#         st.subheader("Banking Overview")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Customers", f"{len(data):,}" if not data.empty else "0")
#         c2.metric("Default Rate", f"{data['default'].mean()*100:.2f}%" if not data.empty else "N/A")
#         c3.metric("Avg Bureau", f"{data['bureau_score'].mean():.0f}" if not data.empty else "N/A")
#         c4.metric("KYC Approved", f"{(data['kyc_status'] == 'Approved').mean()*100:.2f}%" if not data.empty else "N/A")

#         if not data.empty:
#             left, right = st.columns(2)
#             with left:
#                 fig = px.histogram(data, x="bureau_score", color="loan_status", nbins=30, title="Bureau Score Distribution")
#                 st.plotly_chart(fig, use_container_width=True)
#             with right:
#                 fig = px.scatter(data.sample(min(3000, len(data))), x="dti", y="bureau_score", color="default", trendline="ols", title="DTI vs Bureau Score")
#                 st.plotly_chart(fig, use_container_width=True)

#             fig = px.bar(data.groupby("employment_type")["default"].mean().reset_index(), x="employment_type", y="default", title="Default Rate by Employment")
#             st.plotly_chart(fig, use_container_width=True)

#     elif menu == "KYC Verification":
#         st.subheader("KYC Verification")
#         with st.form("kyc_form"):
#             c1, c2, c3 = st.columns(3)
#             with c1:
#                 doc_type = st.selectbox("Document Type", ["PAN", "Aadhaar", "Passport", "Voter ID"])
#                 doc_status = st.selectbox("Document Status", ["Verified", "Mismatch", "Not Provided"])
#                 address_status = st.selectbox("Address Status", ["Verified", "Mismatch", "Not Provided"])
#             with c2:
#                 selfie_score = st.slider("Selfie Match Score", 0.0, 1.0, 0.9, 0.01)
#                 sanction_flag = st.checkbox("Sanctions Hit")
#                 pep_flag = st.checkbox("PEP Flag")
#             with c3:
#                 completion = st.selectbox("KYC Completion", ["Complete", "Partial", "Pending"])
#                 customer_name = st.text_input("Customer Name")
#                 customer_id = st.text_input("Customer ID")
#             submit = st.form_submit_button("Run KYC Check")

#         if submit:
#             tier, status, risk = kyc_assess(doc_type, doc_status, address_status, selfie_score, sanction_flag, pep_flag, completion)
#             st.metric("KYC Tier", tier)
#             st.metric("KYC Status", status)
#             st.metric("KYC Risk", f"{risk:.3f}")
#             if status == "Approved":
#                 st.success("Customer cleared for onboarding.")
#             elif status == "Review":
#                 st.warning("Customer requires manual review.")
#             else:
#                 st.error("Customer rejected due to KYC risk.")

#     elif menu == "Credit Scoring":
#         st.subheader("Credit Scoring")
#         if model is None:
#             st.warning("Train the model first by running train_credit_model.py.")
#         with st.form("score_form"):
#             c1, c2, c3 = st.columns(3)
#             with c1:
#                 age = st.slider("Age", 18, 80, 35)
#                 gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#                 marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
#                 education = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Post Graduate", "Doctorate"])
#                 employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Business Owner", "Government", "Retired", "Student", "Unemployed"])
#                 region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
#                 city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"])
#                 residency_type = st.selectbox("Residency", ["Owned", "Rented", "Family", "Mortgaged"])
#             with c2:
#                 annual_income = st.number_input("Annual Income", 0.0, 50000000.0, 800000.0)
#                 monthly_income = st.number_input("Monthly Income", 0.0, 5000000.0, 65000.0)
#                 monthly_expenses = st.number_input("Monthly Expenses", 0.0, 5000000.0, 42000.0)
#                 savings_balance = st.number_input("Savings Balance", 0.0, 50000000.0, 180000.0)
#                 existing_debt = st.number_input("Existing Debt", 0.0, 50000000.0, 120000.0)
#                 loan_amount = st.number_input("Loan Amount", 0.0, 50000000.0, 500000.0)
#                 tenure_months = st.number_input("Tenure Months", 6, 360, 60)
#                 interest_rate = st.number_input("Interest Rate", 0.0, 40.0, 13.0)
#                 emi = st.number_input("EMI", 0.0, 1000000.0, 12000.0)
#             with c3:
#                 credit_history_years = st.slider("Credit History Years", 0.0, 35.0, 5.0)
#                 num_active_accounts = st.number_input("Active Accounts", 0, 20, 4)
#                 num_open_loans = st.number_input("Open Loans", 0, 10, 1)
#                 credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.32)
#                 delinquency_30 = st.number_input("30D Delinq.", 0, 10, 0)
#                 delinquency_60 = st.number_input("60D Delinq.", 0, 10, 0)
#                 delinquency_90 = st.number_input("90D Delinq.", 0, 10, 0)
#                 recent_inquiries = st.number_input("Recent Inquiries", 0, 20, 1)
#                 repayment_ratio = st.slider("Repayment Ratio", 0.0, 1.0, 0.92)
#                 avg_balance = st.number_input("Avg Balance", 0.0, 50000000.0, 220000.0)
#                 txn_count_month = st.number_input("Monthly Txn Count", 0, 1000, 75)
#                 cashflow_stability = st.slider("Cashflow Stability", 0.0, 1.0, 0.72)
#                 salary_credit_flag = st.selectbox("Salary Credit", [0, 1])
#                 overdraft_usage = st.slider("Overdraft Usage", 0.0, 1.0, 0.1)
#                 collateral_flag = st.selectbox("Collateral Flag", [0, 1])
#                 collateral_value = st.number_input("Collateral Value", 0.0, 100000000.0, 0.0)
#                 pan_status = st.selectbox("PAN Status", ["Verified", "Mismatch", "Not Provided"])
#                 address_status = st.selectbox("Address Status", ["Verified", "Mismatch", "Not Provided"])
#                 selfie_match_score = st.slider("Selfie Match", 0.0, 1.0, 0.92)
#                 sanction_flag = st.selectbox("Sanction Hit", [0, 1])
#                 pep_flag = st.selectbox("PEP Flag", [0, 1])
#                 kyc_doc_quality = st.selectbox("KYC Doc Quality", ["High", "Medium", "Low"])
#                 kyc_completion = st.selectbox("KYC Completion", ["Complete", "Partial", "Pending"])
#                 kyc_risk_score = st.slider("KYC Risk Score", 0.0, 1.0, 0.1)
#                 kyc_status = st.selectbox("KYC Status", ["Approved", "Review", "Rejected"])
#                 bureau_score = st.slider("Bureau Score", 300, 900, 710)
#                 credit_score_band = st.selectbox("Credit Score Band", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
#                 dti = st.slider("DTI", 0.0, 1.5, 0.28)
#                 affordability_ratio = st.slider("Affordability Ratio", 0.0, 1.5, 0.62)
#                 approval_status = st.selectbox("Approval Status", ["Approved", "Rejected"])
#                 loan_status = st.selectbox("Loan Status", ["Performing", "At Risk", "Defaulted"])
#                 risk_band = st.selectbox("Risk Band", ["Low", "Medium", "High", "Critical"])
#             run = st.form_submit_button("Predict Risk")

#         if run and model is not None:
#             input_row = build_input_frame(locals())
#             input_row = input_row[[
#                 "age", "gender", "marital_status", "education", "employment_type", "region", "city_tier", "residency_type",
#                 "annual_income", "monthly_income", "monthly_expenses", "savings_balance", "existing_debt", "loan_amount",
#                 "tenure_months", "interest_rate", "emi", "credit_history_years", "num_active_accounts", "num_open_loans",
#                 "credit_utilization", "delinquency_30", "delinquency_60", "delinquency_90", "recent_inquiries", "repayment_ratio",
#                 "avg_balance", "txn_count_month", "cashflow_stability", "salary_credit_flag", "overdraft_usage", "collateral_flag",
#                 "collateral_value", "pan_status", "address_status", "selfie_match_score", "sanction_flag", "pep_flag",
#                 "kyc_doc_quality", "kyc_completion", "kyc_risk_score", "kyc_status", "bureau_score", "credit_score_band",
#                 "dti", "affordability_ratio", "approval_status", "loan_status", "risk_band"
#             ]]
#             prob = float(model.predict_proba(input_row)[0, 1])
#             score = int(max(300, min(900, 850 - prob * 500)))
#             band = "Low" if prob < 0.25 else "Medium" if prob < 0.5 else "High" if prob < 0.75 else "Critical"
#             st.metric("Default Probability", f"{prob:.2%}")
#             st.metric("Estimated Score", score)
#             st.metric("Risk Band", band)
#             if prob < 0.25:
#                 st.success("Low risk customer.")
#             elif prob < 0.5:
#                 st.warning("Medium risk customer.")
#             else:
#                 st.error("High risk customer.")

#     elif menu == "Customer Explorer":
#         st.subheader("Customer Explorer")
#         if scored.empty:
#             st.info("Generate data and train the model first.")
#         else:
#             c1, c2, c3 = st.columns(3)
#             c1.metric("Avg Default Prob.", f"{scored['predicted_default_probability'].mean():.3f}")
#             c2.metric("Approved KYC", f"{(scored['kyc_status'] == 'Approved').mean() * 100:.2f}%")
#             c3.metric("Critical Risk", f"{(scored['predicted_risk_band'] == 'Critical').mean() * 100:.2f}%")
#             f1, f2 = st.columns(2)
#             with f1:
#                 fig = px.histogram(scored, x="bureau_score", color="predicted_risk_band", nbins=35, title="Bureau Score by Risk")
#                 st.plotly_chart(fig, use_container_width=True)
#             with f2:
#                 fig = px.scatter(scored.sample(min(4000, len(scored))), x="dti", y="predicted_default_probability", color="kyc_status", title="DTI vs Default Probability")
#                 st.plotly_chart(fig, use_container_width=True)
#             st.dataframe(scored.head(200), use_container_width=True)
#             st.download_button("Download Scored Data", scored.to_csv(index=False).encode("utf-8"), "synthetic_banking_scored.csv", "text/csv")

#     else:
#         st.subheader("Model Metrics")
#         if not metrics:
#             st.info("Train the model first.")
#         else:
#             mdf = pd.DataFrame(metrics)
#             st.dataframe(mdf, use_container_width=True)
#             fig = px.bar(mdf, x="model", y="roc_auc", color="roc_auc", title="Model ROC AUC")
#             st.plotly_chart(fig, use_container_width=True)


# if __name__ == "__main__":
#     main()

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import sklearn, sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FINEDA_DIR = BASE_DIR / "Financial EDA & Credit Scoring"
OUT = FINEDA_DIR / "output"

DATA_PATH = OUT / "synthetic_banking_data.csv"
SCORING_PATH = OUT / "synthetic_banking_scored.csv"
KYC_PATH = OUT / "synthetic_kyc_data.csv"
MODEL_PATH = OUT / "credit_scoring_model.joblib"
METRICS_PATH = OUT / "credit_model_metrics.json"
FEATURES_PATH = OUT / "feature_columns.json"

# st.set_page_config(page_title="Financial EDA & Credit Scoring", page_icon="🏦", layout="wide")

def load_assets():
    data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    scored = pd.read_csv(SCORING_PATH) if SCORING_PATH.exists() else pd.DataFrame()
    kyc = pd.read_csv(KYC_PATH) if KYC_PATH.exists() else pd.DataFrame()
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    feature_cols = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else []
    return data, scored, kyc, model, metrics, feature_cols

def add_manual_trendline(fig, df, x_col, y_col, color="red"):
    d = df[[x_col, y_col]].dropna()
    if len(d) < 2:
        return fig
    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()
    m, b = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ys = m * xs + b
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Trendline", line=dict(color=color, width=3)))
    return fig

def align_input_to_training(input_df, feature_cols):
    aligned = input_df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    extra_cols = [c for c in aligned.columns if c not in feature_cols]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)
    return aligned[feature_cols]

def score_customer(model, input_df, feature_cols):
    X = align_input_to_training(input_df, feature_cols)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    if hasattr(model, "decision_function"):
        return float(model.decision_function(X)[0])
    return float(model.predict(X)[0])

def kyc_assess(doc_quality, completion, selfie_score, sanction_flag, pep_flag):
    risk = 0
    risk += 1.0 if doc_quality == "Poor" else 0.4 if doc_quality == "Average" else 0
    risk += 0.8 if completion != "Complete" else 0
    risk += 1.4 if sanction_flag else 0
    risk += 1.2 if pep_flag else 0
    risk += 0.9 if selfie_score < 0.75 else 0
    if risk >= 2.5:
        return "High", "Rejected", round(min(risk / 5, 1), 3)
    if risk >= 1.2:
        return "Medium", "Review", round(min(risk / 5, 1), 3)
    return "Low", "Approved", round(min(risk / 5, 1), 3)

def main():
    print("python:", sys.version)
    print("sklearn:", sklearn.__version__)
    data, scored, kyc, model, metrics, feature_cols = load_assets()
    st.title("Financial EDA & Credit Scoring")
    st.caption("Credit scoring app with model scoring, KYC, and analytics.")

    menu = st.sidebar.radio("Modules", ["Dashboard", "Credit Scoring", "KYC Verification", "Customer Explorer", "Model Metrics"])

    if menu == "Dashboard":
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{len(data):,}" if not data.empty else "0")
        c2.metric("Default Rate", f"{data['default'].mean()*100:.2f}%" if not data.empty and "default" in data.columns else "N/A")
        c3.metric("Avg Bureau Score", f"{data['bureau_score'].mean():.0f}" if not data.empty and "bureau_score" in data.columns else "N/A")
        c4.metric("Approved KYC", f"{(data['kyc_status'] == 'Approved').mean()*100:.2f}%" if not data.empty and "kyc_status" in data.columns else "N/A")

        if not data.empty:
            left, right = st.columns(2)
            with left:
                fig = px.histogram(data, x="bureau_score", color="loan_status", nbins=30, title="Bureau Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                sample_df = data.sample(min(3000, len(data)), random_state=42)
                fig = px.scatter(sample_df, x="dti", y="bureau_score", color="default", title="DTI vs Bureau Score")
                fig = add_manual_trendline(fig, sample_df, "dti", "bureau_score")
                st.plotly_chart(fig, use_container_width=True)

    elif menu == "Credit Scoring":
        st.subheader("Credit Scoring")
        if model is None or not feature_cols:
            st.warning("Train the model first.")
        else:
            with st.form("score_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.slider("Age", 21, 65, 35)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                    education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "PhD"])
                    region = st.selectbox("Region", ["North", "South", "East", "West"])
                    city_tier = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])
                    residency_type = st.selectbox("Residency Type", ["Own", "Rent", "Family"])
                with c2:
                    annual_income = st.number_input("Annual Income", min_value=18000.0, value=50000.0, step=1000.0)
                    monthly_income = st.number_input("Monthly Income", min_value=1000.0, value=4167.0, step=100.0)
                    monthly_expenses = st.number_input("Monthly Expenses", min_value=500.0, value=2500.0, step=100.0)
                    savings_balance = st.number_input("Savings Balance", min_value=0.0, value=10000.0, step=500.0)
                    existing_debt = st.number_input("Existing Debt", min_value=0.0, value=5000.0, step=500.0)
                    loan_amount = st.number_input("Loan Amount", min_value=1000.0, value=15000.0, step=500.0)
                    tenure_months = st.slider("Tenure Months", 6, 84, 24)
                    interest_rate = st.number_input("Interest Rate", min_value=5.0, value=10.0, step=0.1)
                with c3:
                    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Contract", "Unemployed", "Student"])
                    credit_history_years = st.slider("Credit History Years", 0, 30, 5)
                    num_active_accounts = st.slider("Active Accounts", 1, 12, 4)
                    num_open_loans = st.slider("Open Loans", 0, 5, 1)
                    credit_utilization = st.slider("Credit Utilization", 0.01, 0.99, 0.35, 0.01)
                    delinquency_30 = st.slider("30D Delinquency", 0, 10, 0)
                    delinquency_60 = st.slider("60D Delinquency", 0, 10, 0)
                    delinquency_90 = st.slider("90D Delinquency", 0, 10, 0)
                    recent_inquiries = st.slider("Recent Inquiries", 0, 10, 1)
                    cashflow_stability = st.slider("Cashflow Stability", 0.05, 0.99, 0.72, 0.01)
                    salary_credit_flag = st.checkbox("Salary Credit Flag", value=True)
                submit = st.form_submit_button("Score Customer")

            if submit:
                dti = (existing_debt + loan_amount) / max(annual_income, 1)
                repayment_ratio = monthly_income / max(monthly_expenses + interest_rate, 1)
                avg_balance = savings_balance * 1.1
                txn_count_month = 40
                overdraft_usage = 0
                collateral_flag = 0
                collateral_value = 0
                kyc_doc_quality = "Good"
                kyc_completion = "Complete"
                affinity = (monthly_income - monthly_expenses) / max(monthly_income, 1)

                row = {
                    "age": age,
                    "gender": gender,
                    "marital_status": marital_status,
                    "education": education,
                    "employment_type": employment_type,
                    "region": region,
                    "city_tier": city_tier,
                    "residency_type": residency_type,
                    "annual_income": annual_income,
                    "monthly_income": monthly_income,
                    "monthly_expenses": monthly_expenses,
                    "savings_balance": savings_balance,
                    "existing_debt": existing_debt,
                    "loan_amount": loan_amount,
                    "tenure_months": tenure_months,
                    "interest_rate": interest_rate,
                    "credit_history_years": credit_history_years,
                    "num_active_accounts": num_active_accounts,
                    "num_open_loans": num_open_loans,
                    "credit_utilization": credit_utilization,
                    "delinquency_30": delinquency_30,
                    "delinquency_60": delinquency_60,
                    "delinquency_90": delinquency_90,
                    "recent_inquiries": recent_inquiries,
                    "repayment_ratio": repayment_ratio,
                    "avg_balance": avg_balance,
                    "txn_count_month": txn_count_month,
                    "cashflow_stability": cashflow_stability,
                    "salary_credit_flag": int(salary_credit_flag),
                    "overdraft_usage": overdraft_usage,
                    "collateral_flag": collateral_flag,
                    "collateral_value": collateral_value,
                    "kyc_doc_quality": kyc_doc_quality,
                    "kyc_completion": kyc_completion,
                    "bureau_score": 680,
                    "dti": dti,
                    "affordability_ratio": affinity,
                }

                input_df = pd.DataFrame([row])
                try:
                    score = score_customer(model, input_df, feature_cols)
                    risk_level = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
                    decision = "Reject" if score >= 0.7 else "Review" if score >= 0.4 else "Approve"
                    st.metric("Default Score", f"{score:.3f}")
                    st.metric("Risk Level", risk_level)
                    st.metric("Decision", decision)
                except Exception as e:
                    st.error(f"Scoring failed: {e}")

    elif menu == "KYC Verification":
        st.subheader("KYC Verification")
        with st.form("kyc_form"):
            c1, c2 = st.columns(2)
            with c1:
                doc_quality = st.selectbox("Document Quality", ["Good", "Average", "Poor"])
                completion = st.selectbox("Completion", ["Complete", "Partial", "Pending"])
            with c2:
                selfie_score = st.slider("Selfie Match Score", 0.0, 1.0, 0.9, 0.01)
                sanction_flag = st.checkbox("Sanctions Hit")
                pep_flag = st.checkbox("PEP Flag")
            submit = st.form_submit_button("Run KYC")
        if submit:
            tier, status, risk = kyc_assess(doc_quality, completion, selfie_score, sanction_flag, pep_flag)
            st.metric("KYC Tier", tier)
            st.metric("KYC Status", status)
            st.metric("KYC Risk", f"{risk:.3f}")

    elif menu == "Customer Explorer":
        st.subheader("Customer Explorer")
        if not data.empty:
            st.dataframe(data.head(200), use_container_width=True)
        else:
            st.info("No data loaded yet.")

    elif menu == "Model Metrics":
        st.subheader("Model Metrics")
        if not metrics:
            st.info("Train the model first.")
        else:
            mdf = pd.DataFrame(metrics["metrics"])
            st.dataframe(mdf, use_container_width=True)
            fig = px.bar(mdf, x="model", y="roc_auc", color="roc_auc", title="ROC AUC by Model")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()