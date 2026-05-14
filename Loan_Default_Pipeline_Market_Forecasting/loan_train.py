import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE_DIR / "output"
OUT.mkdir(exist_ok=True)

DATA_PATH = OUT / "synthetic_loan_data.csv"
MODEL_PATH = OUT / "loan_default_gb_model.joblib"
FEATURES_PATH = OUT / "loan_feature_columns.json"
METRICS_PATH = OUT / "loan_metrics.json"
SCORED_PATH = OUT / "loan_scored_data.csv"
CM_PATH = OUT / "loan_confusion_matrix.png"


def generate_loan_synthetic_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.clip(rng.lognormal(mean=11.0, sigma=0.55, size=n), 5000, 5000000).round(0)
    loan_amount = np.clip(rng.lognormal(mean=10.0, sigma=0.75, size=n), 10000, 2000000).round(0)
    credit_score = np.clip(rng.normal(690, 55, n), 300, 850).round(0)
    dti = np.clip(rng.beta(2.2, 5.0, n) * 0.9, 0.01, 0.95).round(3)
    delinquency = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.70, 0.14, 0.08, 0.04, 0.02, 0.02])
    inquiries = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.20, 0.20, 0.18, 0.15, 0.11, 0.09, 0.07])
    employment_years = np.clip(rng.gamma(2.5, 3.0, n), 0, 40).round(0)
    savings = np.clip(rng.lognormal(mean=10.3, sigma=0.9, size=n), 0, 4000000).round(0)
    open_accounts = rng.integers(1, 14, n)
    loan_tenure = np.clip(rng.gamma(2.0, 10.0, n), 1, 360).round(0)
    interest_rate = np.clip(rng.normal(11.5, 2.8, n), 5.0, 28.0).round(2)

    emp = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n, p=[0.56, 0.16, 0.20, 0.08])
    home = rng.choice(["Own", "Rent", "Mortgage"], n, p=[0.30, 0.48, 0.22])
    purpose = rng.choice(["Debt consolidation", "Home improvement", "Business", "Medical", "Education", "Other"], n,
                         p=[0.28, 0.14, 0.14, 0.16, 0.10, 0.18])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.22, 0.20, 0.19, 0.20, 0.19])

    risk_score = (-3.0 + 0.018 * (650 - credit_score) + 0.80 * delinquency + 0.50 * inquiries + 2.0 * dti +
                  0.6 * (loan_amount / np.maximum(income, 1)) - 0.00000025 * income +
                  np.where(emp == "Unemployed", 0.7, 0) + np.where(home == "Rent", 0.2, 0) + rng.normal(0, 0.55, n))
    prob = 1 / (1 + np.exp(-risk_score))
    default = (prob > np.quantile(prob, 0.84)).astype(int)

    return pd.DataFrame({
        "age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "credit_history_years": np.clip(rng.gamma(2.1, 4.0, n), 0, 40).round(0),
        "debt_to_income_ratio": dti,
        "num_delinquencies": delinquency,
        "num_open_accounts": open_accounts,
        "recent_credit_inquiries": inquiries,
        "savings_balance": savings,
        "employment_years": employment_years,
        "interest_rate": interest_rate,
        "loan_tenure_months": loan_tenure,
        "employment_status": emp,
        "rent_or_own": home,
        "loan_purpose": purpose,
        "region": region,
        "default": default,
    })


def main():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_loan_synthetic_data()
        df.to_csv(DATA_PATH, index=False)

    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        mode = X[c].mode(dropna=True)
        X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    feature_cols = X_enc.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    joblib.dump(model, MODEL_PATH)
    json.dump(feature_cols, open(FEATURES_PATH, "w"), indent=2)
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Loan Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=220)
    plt.close()

    scored = df.copy()
    scored["default_probability"] = model.predict_proba(X_enc)[:, 1]
    scored["expected_loss"] = scored["default_probability"] * 0.45 * scored["loan_amount"]
    scored.to_csv(SCORED_PATH, index=False)
    print(metrics)


if __name__ == "__main__":
    main()