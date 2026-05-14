# """
# Synthetic banking dataset generator.
# Logic:
# - Create realistic customer, KYC, financial, bureau, banking behavior, and loan fields.
# - Inject controlled correlations so credit score, default, and KYC outcomes are learnable.
# - Save clean training datasets for EDA and model development.
# """

# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd

# BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# OUT = BASE_DIR / "output"
# OUT.mkdir(exist_ok=True)

# DATA_PATH = OUT / "synthetic_banking_data.csv"
# KYC_PATH = OUT / "synthetic_kyc_data.csv"
# META_PATH = OUT / "synthetic_data_profile.json"


# def _soft_clip(x, lo, hi):
#     return np.clip(x, lo, hi)


# def generate_synthetic_banking_data(n=25000, seed=42):
#     rng = np.random.default_rng(seed)

#     customer_id = [f"CUST{str(i).zfill(7)}" for i in range(1, n + 1)]
#     age = rng.integers(21, 76, n)
#     gender = rng.choice(["Male", "Female", "Other"], n, p=[0.58, 0.40, 0.02])
#     marital_status = rng.choice(["Single", "Married", "Divorced", "Widowed"], n, p=[0.30, 0.57, 0.10, 0.03])
#     education = rng.choice(["High School", "Diploma", "Graduate", "Post Graduate", "Doctorate"], n, p=[0.16, 0.20, 0.34, 0.24, 0.06])
#     employment_type = rng.choice(["Salaried", "Self-employed", "Business Owner", "Government", "Retired", "Student", "Unemployed"], n,
#                                  p=[0.46, 0.16, 0.10, 0.12, 0.07, 0.04, 0.05])
#     region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.23, 0.21, 0.19, 0.20, 0.17])
#     city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.31, 0.45, 0.24])
#     residency_type = rng.choice(["Owned", "Rented", "Family", "Mortgaged"], n, p=[0.31, 0.38, 0.20, 0.11])

#     annual_income = _soft_clip(rng.lognormal(mean=13.1, sigma=0.55, size=n), 180000, 15000000).round(0)
#     monthly_income = (annual_income / 12).round(0)
#     monthly_expenses = _soft_clip(monthly_income * rng.normal(0.58, 0.12, n), 10000, None).round(0)
#     savings_balance = _soft_clip(rng.lognormal(mean=11.2, sigma=1.0, size=n), 0, 12000000).round(0)
#     existing_debt = _soft_clip(annual_income * rng.normal(0.18, 0.13, n), 0, 8000000).round(0)
#     loan_amount = _soft_clip(rng.lognormal(mean=12.0, sigma=0.7, size=n), 50000, 25000000).round(0)
#     tenure_months = rng.integers(6, 361, n)
#     interest_rate = _soft_clip(rng.normal(12.5, 3.8, n), 6.5, 32.0).round(2)
#     emi = _soft_clip((loan_amount * (interest_rate / 12 / 100)) / (1 - (1 + interest_rate / 12 / 100) ** (-tenure_months)), 1000, 1000000).round(0)

#     credit_history_years = _soft_clip(rng.gamma(2.4, 3.0, n), 0, 35).round(1)
#     num_active_accounts = rng.integers(1, 12, n)
#     num_open_loans = rng.integers(0, 6, n)
#     credit_utilization = _soft_clip(rng.beta(2.1, 4.8, n), 0.01, 0.99).round(3)
#     delinquency_30 = rng.choice([0, 1, 2, 3], n, p=[0.78, 0.16, 0.05, 0.01])
#     delinquency_60 = rng.choice([0, 1, 2], n, p=[0.88, 0.10, 0.02])
#     delinquency_90 = rng.choice([0, 1], n, p=[0.95, 0.05])
#     recent_inquiries = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.24, 0.21, 0.18, 0.14, 0.10, 0.08, 0.05])
#     repayment_ratio = _soft_clip(rng.normal(0.92, 0.08, n), 0.40, 1.0).round(3)
#     avg_balance = _soft_clip(rng.lognormal(mean=10.7, sigma=0.9, size=n), 0, 9000000).round(0)
#     txn_count_month = rng.integers(5, 220, n)
#     cashflow_stability = _soft_clip(rng.normal(0.68, 0.16, n), 0.05, 1.0).round(3)
#     salary_credit_flag = rng.choice([0, 1], n, p=[0.28, 0.72])
#     overdraft_usage = _soft_clip(rng.beta(1.5, 7.5, n), 0, 1).round(3)
#     collateral_flag = rng.choice([0, 1], n, p=[0.63, 0.37])
#     collateral_value = np.where(collateral_flag == 1, _soft_clip(loan_amount * rng.normal(1.25, 0.40, n), 0, 50000000), 0).round(0)

#     pan_status = rng.choice(["Verified", "Mismatch", "Not Provided"], n, p=[0.89, 0.05, 0.06])
#     address_status = rng.choice(["Verified", "Mismatch", "Not Provided"], n, p=[0.86, 0.08, 0.06])
#     selfie_match_score = _soft_clip(rng.normal(0.91, 0.09, n), 0.05, 1.0).round(3)
#     sanction_flag = rng.choice([0, 1], n, p=[0.985, 0.015])
#     pep_flag = rng.choice([0, 1], n, p=[0.98, 0.02])
#     kyc_doc_quality = rng.choice(["High", "Medium", "Low"], n, p=[0.72, 0.22, 0.06])
#     kyc_completion = rng.choice(["Complete", "Partial", "Pending"], n, p=[0.84, 0.11, 0.05])

#     kyc_risk = (
#         1.4 * (pan_status != "Verified").astype(int)
#         + 1.2 * (address_status != "Verified").astype(int)
#         + 1.7 * sanction_flag
#         + 1.3 * pep_flag
#         + 1.0 * (kyc_doc_quality == "Low").astype(int)
#         + 0.8 * (kyc_completion != "Complete").astype(int)
#         + rng.normal(0, 0.45, n)
#     )
#     kyc_risk_score = _soft_clip((kyc_risk - kyc_risk.min()) / (kyc_risk.max() - kyc_risk.min() + 1e-9), 0, 1).round(3)
#     kyc_status = np.where((sanction_flag == 1) | (pep_flag == 1) | (kyc_risk_score > 0.72), "Rejected",
#                    np.where(kyc_risk_score > 0.45, "Review", "Approved"))

#     bureau_score = (
#         820
#         - 85 * credit_utilization
#         - 38 * delinquency_30
#         - 52 * delinquency_60
#         - 72 * delinquency_90
#         - 12 * recent_inquiries
#         + 9 * credit_history_years
#         + 14 * repayment_ratio
#         + 10 * cashflow_stability
#         + rng.normal(0, 28, n)
#     )
#     bureau_score = _soft_clip(bureau_score, 300, 900).round(0)

#     dti = _soft_clip((existing_debt + emi * tenure_months / 12) / np.maximum(annual_income, 1), 0.01, 1.5).round(3)
#     affordability_ratio = _soft_clip(monthly_expenses / np.maximum(monthly_income, 1), 0.1, 1.4).round(3)

#     default_risk = (
#         -4.0
#         + 0.018 * (700 - bureau_score)
#         + 2.7 * dti
#         + 1.6 * credit_utilization
#         + 0.9 * delinquency_30
#         + 1.2 * delinquency_60
#         + 1.8 * delinquency_90
#         + 0.55 * recent_inquiries
#         - 0.0025 * credit_history_years * 10
#         - 0.7 * repayment_ratio
#         - 0.9 * cashflow_stability
#         + 0.3 * overdraft_usage
#         + 0.8 * (employment_type == "Unemployed").astype(int)
#         + 0.5 * (employment_type == "Student").astype(int)
#         + 0.7 * (salary_credit_flag == 0).astype(int)
#         + 0.6 * (kyc_status == "Review").astype(int)
#         + 1.0 * (kyc_status == "Rejected").astype(int)
#         + rng.normal(0, 0.65, n)
#     )
#     default_probability = 1 / (1 + np.exp(-default_risk))
#     default = (default_probability > np.quantile(default_probability, 0.84)).astype(int)

#     loan_status = np.where(default == 1, "Defaulted", np.where(default_probability > 0.45, "At Risk", "Performing"))
#     approval_status = np.where((kyc_status == "Approved") & (bureau_score >= 620) & (dti <= 0.6) & (default_probability < 0.45), "Approved", "Rejected")
#     risk_band = pd.cut(default_probability, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["Low", "Medium", "High", "Critical"], include_lowest=True)
#     credit_score_band = pd.cut(bureau_score, bins=[0, 579, 669, 739, 799, 900], labels=["Poor", "Fair", "Good", "Very Good", "Excellent"], include_lowest=True)

#     df = pd.DataFrame({
#         "customer_id": customer_id,
#         "age": age,
#         "gender": gender,
#         "marital_status": marital_status,
#         "education": education,
#         "employment_type": employment_type,
#         "region": region,
#         "city_tier": city_tier,
#         "residency_type": residency_type,
#         "annual_income": annual_income,
#         "monthly_income": monthly_income,
#         "monthly_expenses": monthly_expenses,
#         "savings_balance": savings_balance,
#         "existing_debt": existing_debt,
#         "loan_amount": loan_amount,
#         "tenure_months": tenure_months,
#         "interest_rate": interest_rate,
#         "emi": emi,
#         "credit_history_years": credit_history_years,
#         "num_active_accounts": num_active_accounts,
#         "num_open_loans": num_open_loans,
#         "credit_utilization": credit_utilization,
#         "delinquency_30": delinquency_30,
#         "delinquency_60": delinquency_60,
#         "delinquency_90": delinquency_90,
#         "recent_inquiries": recent_inquiries,
#         "repayment_ratio": repayment_ratio,
#         "avg_balance": avg_balance,
#         "txn_count_month": txn_count_month,
#         "cashflow_stability": cashflow_stability,
#         "salary_credit_flag": salary_credit_flag,
#         "overdraft_usage": overdraft_usage,
#         "collateral_flag": collateral_flag,
#         "collateral_value": collateral_value,
#         "pan_status": pan_status,
#         "address_status": address_status,
#         "selfie_match_score": selfie_match_score,
#         "sanction_flag": sanction_flag,
#         "pep_flag": pep_flag,
#         "kyc_doc_quality": kyc_doc_quality,
#         "kyc_completion": kyc_completion,
#         "kyc_risk_score": kyc_risk_score,
#         "kyc_status": kyc_status,
#         "bureau_score": bureau_score,
#         "credit_score_band": credit_score_band.astype(str),
#         "dti": dti,
#         "affordability_ratio": affordability_ratio,
#         "default_probability": default_probability.round(3),
#         "risk_band": risk_band.astype(str),
#         "approval_status": approval_status,
#         "loan_status": loan_status,
#         "default": default,
#     })

#     df.to_csv(DATA_PATH, index=False)
#     kyc_cols = ["customer_id", "pan_status", "address_status", "selfie_match_score", "sanction_flag", "pep_flag", "kyc_doc_quality", "kyc_completion", "kyc_risk_score", "kyc_status"]
#     df[kyc_cols].to_csv(KYC_PATH, index=False)
#     profile = {
#         "rows": int(len(df)),
#         "default_rate": float(df["default"].mean()),
#         "kyc_approved_rate": float((df["kyc_status"] == "Approved").mean()),
#         "avg_bureau_score": float(df["bureau_score"].mean()),
#         "avg_income": float(df["annual_income"].mean()),
#     }
#     json.dump(profile, open(META_PATH, "w"), indent=2)
#     return df


# if __name__ == "__main__":
#     df = generate_synthetic_banking_data()
#     print(df.head())

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE_DIR / "output"
OUT.mkdir(exist_ok=True, parents=True)

DATA_PATH = OUT / "synthetic_banking_data.csv"
KYC_PATH = OUT / "synthetic_kyc_data.csv"
META_PATH = OUT / "synthetic_banking_meta.json"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_synthetic_banking_data(n=20000, random_state=42):
    rng = np.random.default_rng(random_state)

    customer_id = [f"CUST{i:06d}" for i in range(1, n + 1)]
    age = rng.integers(21, 66, n)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.49, 0.49, 0.02])
    marital_status = rng.choice(["Single", "Married", "Divorced"], n, p=[0.42, 0.50, 0.08])
    education = rng.choice(["High School", "Associate", "Bachelor", "Master", "PhD"], n, p=[0.20, 0.22, 0.34, 0.19, 0.05])
    employment_type = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed", "Student"], n, p=[0.50, 0.18, 0.17, 0.10, 0.05])
    region = rng.choice(["North", "South", "East", "West"], n)
    city_tier = rng.choice(["Tier1", "Tier2", "Tier3"], n, p=[0.25, 0.40, 0.35])
    residency_type = rng.choice(["Own", "Rent", "Family"], n, p=[0.38, 0.47, 0.15])

    annual_income = np.clip(rng.lognormal(mean=11.0, sigma=0.45, size=n), 18000, 250000).round(0)
    monthly_income = annual_income / 12
    monthly_expenses = np.clip(monthly_income * rng.uniform(0.45, 0.85, n), 800, None)
    savings_balance = np.clip(annual_income * rng.uniform(0.02, 0.35, n), 0, None)
    existing_debt = np.clip(annual_income * rng.uniform(0.0, 0.65, n), 0, None)
    loan_amount = np.clip(annual_income * rng.uniform(0.08, 0.9, n), 2000, 200000)
    tenure_months = rng.integers(6, 84, n)
    interest_rate = np.clip(rng.normal(10.0, 2.8, n), 5.0, 22.0).round(2)
    credit_history_years = rng.integers(0, 31, n)
    num_active_accounts = rng.integers(1, 13, n)
    num_open_loans = rng.integers(0, 5, n)
    credit_utilization = np.clip(rng.beta(2.5, 5.5, n), 0.02, 0.98).round(3)
    delinquency_30 = rng.poisson(0.35, n)
    delinquency_60 = rng.poisson(0.15, n)
    delinquency_90 = rng.poisson(0.07, n)
    recent_inquiries = rng.poisson(1.5, n)
    avg_balance = np.clip(savings_balance * rng.uniform(0.5, 1.5, n), 0, None)
    txn_count_month = rng.integers(8, 151, n)
    cashflow_stability = np.clip(rng.normal(0.72, 0.14, n), 0.05, 0.99).round(3)
    salary_credit_flag = rng.binomial(1, 0.72, n)
    overdraft_usage = rng.poisson(0.4, n)
    collateral_flag = rng.binomial(1, 0.38, n)
    collateral_value = np.where(collateral_flag == 1, np.clip(loan_amount * rng.uniform(0.8, 2.5, n), 0, None), 0).round(0)

    kyc_doc_quality = rng.choice(["Good", "Average", "Poor"], n, p=[0.72, 0.20, 0.08])
    selfie_match_score = np.clip(rng.normal(0.89, 0.08, n), 0.3, 1.0).round(3)
    sanction_flag = rng.binomial(1, 0.03, n)
    pep_flag = rng.binomial(1, 0.02, n)
    kyc_completion = rng.choice(["Complete", "Partial", "Pending"], n, p=[0.82, 0.13, 0.05])

    base_kyc_risk = (
        0.9 * (kyc_doc_quality == "Poor").astype(int)
        + 0.5 * (kyc_doc_quality == "Average").astype(int)
        + 1.2 * (kyc_completion != "Complete").astype(int)
        + 1.7 * sanction_flag
        + 1.4 * pep_flag
        + 1.0 * (selfie_match_score < 0.75).astype(int)
        + rng.normal(0, 0.3, n)
    )
    kyc_risk_score = sigmoid(base_kyc_risk)
    kyc_status = np.where(kyc_risk_score < 0.35, "Approved", np.where(kyc_risk_score < 0.65, "Review", "Rejected"))

    bureau_base = (
        760
        - 42 * credit_utilization
        - 18 * delinquency_30
        - 28 * delinquency_60
        - 40 * delinquency_90
        - 6 * recent_inquiries
        - 22 * (employment_type == "Unemployed").astype(int)
        - 14 * (employment_type == "Student").astype(int)
        + 7 * credit_history_years
        + 0.015 * annual_income / 1000
        + rng.normal(0, 25, n)
    )
    bureau_score = np.clip(bureau_base, 300, 850).round(0)

    dti = np.clip((existing_debt + loan_amount) / (annual_income + 1e-6), 0.01, 2.5).round(3)
    affordability_ratio = np.clip((monthly_income - monthly_expenses) / (monthly_income + 1e-6), -1.0, 1.0).round(3)
    repayment_ratio = np.clip(monthly_income / (monthly_expenses + interest_rate + 1e-6), 0.1, 6.0).round(3)

    default_risk = (
        -4.0
        + 2.7 * dti
        + 1.4 * credit_utilization
        + 0.7 * delinquency_30
        + 1.2 * delinquency_60
        + 1.7 * delinquency_90
        + 0.45 * recent_inquiries
        - 0.004 * credit_history_years
        - 0.8 * cashflow_stability
        + 0.75 * (employment_type == "Unemployed").astype(int)
        + 0.45 * (employment_type == "Student").astype(int)
        + 0.35 * (salary_credit_flag == 0).astype(int)
        + 0.9 * (kyc_status == "Rejected").astype(int)
        + 0.4 * (kyc_status == "Review").astype(int)
        + 0.02 * (700 - bureau_score) / 10
        + rng.normal(0, 1.2, n)
    )
    default_probability = sigmoid(default_risk)
    threshold = np.quantile(default_probability, 0.84)
    default = (default_probability > threshold).astype(int)

    loan_status = np.where(default == 1, "Defaulted", np.where(default_probability > 0.45, "At Risk", "Performing"))
    approval_status = np.where(
        (kyc_status == "Approved") & (bureau_score >= 620) & (dti <= 0.75) & (default_probability < 0.45),
        "Approved",
        "Rejected"
    )
    risk_band = pd.cut(default_probability, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["Low", "Medium", "High", "Critical"], include_lowest=True)
    credit_score_band = pd.cut(bureau_score, bins=[0, 579, 669, 739, 799, 900], labels=["Poor", "Fair", "Good", "Very Good", "Excellent"], include_lowest=True)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "region": region,
        "city_tier": city_tier,
        "residency_type": residency_type,
        "annual_income": annual_income,
        "monthly_income": monthly_income.round(2),
        "monthly_expenses": monthly_expenses.round(2),
        "savings_balance": savings_balance.round(2),
        "existing_debt": existing_debt.round(2),
        "loan_amount": loan_amount.round(2),
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
        "avg_balance": avg_balance.round(2),
        "txn_count_month": txn_count_month,
        "cashflow_stability": cashflow_stability,
        "salary_credit_flag": salary_credit_flag,
        "overdraft_usage": overdraft_usage,
        "collateral_flag": collateral_flag,
        "collateral_value": collateral_value,
        "kyc_doc_quality": kyc_doc_quality,
        "kyc_completion": kyc_completion,
        "kyc_risk_score": kyc_risk_score.round(3),
        "kyc_status": kyc_status,
        "bureau_score": bureau_score,
        "credit_score_band": credit_score_band.astype(str),
        "dti": dti,
        "affordability_ratio": affordability_ratio,
        "default_probability": default_probability.round(3),
        "risk_band": risk_band.astype(str),
        "approval_status": approval_status,
        "loan_status": loan_status,
        "default": default,
    })

    df.to_csv(DATA_PATH, index=False)

    pd.DataFrame({
        "customer_id": customer_id,
        "kyc_doc_quality": kyc_doc_quality,
        "kyc_completion": kyc_completion,
        "kyc_risk_score": kyc_risk_score.round(3),
        "kyc_status": kyc_status,
        "sanction_flag": sanction_flag,
        "pep_flag": pep_flag,
        "selfie_match_score": selfie_match_score,
        "approval_status": approval_status
    }).to_csv(KYC_PATH, index=False)

    META_PATH.write_text(json.dumps({
        "rows": int(len(df)),
        "default_rate": float(df["default"].mean()),
        "kyc_approved_rate": float((df["kyc_status"] == "Approved").mean()),
        "avg_bureau_score": float(df["bureau_score"].mean()),
        "avg_income": float(df["annual_income"].mean())
    }, indent=2))
    return df

if __name__ == "__main__":
    df = generate_synthetic_banking_data()
    print(df.head())
    print(df["default"].value_counts(normalize=True))