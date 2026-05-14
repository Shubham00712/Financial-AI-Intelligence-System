# %%
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# %%
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE_DIR / "output"
OUT.mkdir(exist_ok=True, parents=True)

DATA_PATH = OUT / "synthetic_banking_data.csv"
MODEL_PATH = OUT / "credit_scoring_model.joblib"
METRICS_PATH = OUT / "credit_model_metrics.json"
SCORING_PATH = OUT / "synthetic_banking_scored.csv"
FEATURES_PATH = OUT / "feature_columns.json"

df = pd.read_csv(DATA_PATH)

target = "default"
leak_cols = [
    "default", "default_probability", "risk_band", "approval_status",
    "loan_status", "credit_score_band", "kyc_status", "kyc_risk_score"
]

X = df.drop(columns=[c for c in leak_cols if c in df.columns])
y = df[target].astype(int)

feature_cols = X.columns.tolist()
FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2))


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()


# %%
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

models = {
    "logistic_regression": LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
    "hgb": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        early_stopping=True,
        random_state=42
    )
}


# %%
results = []
best_model = None
best_name = None
best_auc = -1

for name, clf in models.items():
    if name == "hgb":
        hgb_preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ], remainder="drop")
        pipe = Pipeline([("preprocessor", hgb_preprocessor), ("classifier", clf)])
    else:
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        proba = pipe.decision_function(X_test)

    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    results.append(row)

    if row["roc_auc"] > best_auc:
        best_auc = row["roc_auc"]
        best_model = pipe
        best_name = name

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
results_df.to_csv(OUT / "credit_model_metrics.csv", index=False)


# %%
joblib.dump(best_model, MODEL_PATH)

scored = df.copy()
scored_X = df[X.columns].copy()
if hasattr(best_model, "predict_proba"):
    scored["default_score"] = best_model.predict_proba(scored_X)[:, 1]
else:
    scored["default_score"] = best_model.decision_function(scored_X)
scored["default_pred"] = (scored["default_score"] >= 0.5).astype(int)
scored.to_csv(SCORING_PATH, index=False)

METRICS_PATH.write_text(json.dumps({
    "best_model": best_name,
    "best_roc_auc": float(best_auc),
    "rows": int(len(df)),
    "default_rate": float(y.mean()),
    "metrics": results
}, indent=2))

print(results_df)
print(f"Best model: {best_name}")

