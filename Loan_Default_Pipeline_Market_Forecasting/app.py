import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT = BASE_DIR / "Loan_Default_Pipeline_Market_Forecasting" / "output"
LOAN_MODEL_PATH = OUT / "loan_default_gb_model.joblib"
LOAN_FEATURES_PATH = OUT / "loan_feature_columns.json"
LOAN_METRICS_PATH = OUT / "loan_metrics.json"
LOAN_SCORE_PATH = OUT / "loan_scored_data.csv"
MARKET_MODEL_PATH = OUT / "stock_lstm_model.keras"
MARKET_SCALER_PATH = OUT / "stock_scaler.joblib"
MARKET_METRICS_PATH = OUT / "stock_metrics.json"
MARKET_FORECAST_PATH = OUT / "stock_forecast_output.csv"
LOAN_DATA_PATH = OUT / "synthetic_loan_data.csv"
STOCK_DATA_PATH = OUT / "synthetic_stock_data.csv"
STOCK_UNIVERSE_PATH = OUT / "synthetic_stock_universe.csv"
RECO_PATH = OUT / "stock_recommendations.csv"

# st.set_page_config(page_title="Financial Risk Intelligence Suite", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #06111d 0%, #0a1930 45%, #102b45 100%); color: #edf2f7;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #07101b 0%, #0b1829 100%); border-right: 1px solid rgba(255,255,255,0.08);}
.main .block-container {padding-top: 1rem; padding-bottom: 1.8rem;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
div[data-testid="metric-container"] {background: rgba(10, 22, 38, 0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 12px 14px; box-shadow: 0 12px 28px rgba(0,0,0,0.22);}
.stButton button {background: linear-gradient(135deg, #22c55e, #16a34a); color: white; border: 0; border-radius: 12px; padding: 0.55rem 1rem; font-weight: 600;}
.stButton button:hover {background: linear-gradient(135deg, #34d399, #15803d);}
[data-testid="stForm"] {background: rgba(8, 17, 31, 0.90); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 18px;}
h1, h2, h3, h4, p, label, span {color: #edf2f7;}
.section-box {background: rgba(8, 17, 31, 0.84); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px;}
</style>
""", unsafe_allow_html=True)

def ensure_synthetic_stock_universe():
    if STOCK_UNIVERSE_PATH.exists():
        return pd.read_csv(STOCK_UNIVERSE_PATH)
    rng = np.random.default_rng(7)
    stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "LT.NS", "SBIN.NS", "ITC.NS", "TITAN.NS", "AXISBANK.NS"]
    sectors = ["Energy", "IT", "Banking", "IT", "Banking", "Construction", "Banking", "FMCG", "Consumer", "Banking"]
    beta = [1.15, 0.95, 1.05, 0.90, 1.00, 1.10, 1.20, 0.85, 0.98, 1.08]
    df = pd.DataFrame({"symbol": stocks, "sector": sectors, "beta": beta, "momentum": np.round(rng.uniform(0.2, 0.95, len(stocks)), 3), "valuation": np.round(rng.uniform(0.15, 0.85, len(stocks)), 3), "dividend_yield": np.round(rng.uniform(0.01, 0.035, len(stocks)), 4), "pe_ratio": np.round(rng.uniform(12, 45, len(stocks)), 2), "de_ratio": np.round(rng.uniform(0.05, 1.8, len(stocks)), 2)})
    df.to_csv(STOCK_UNIVERSE_PATH, index=False)
    return df

def generate_stock_history(symbol, n_days=1500, seed=42):
    seed_val = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed + seed_val)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    drift_map = {"RELIANCE.NS": 0.0005, "TCS.NS": 0.00045, "HDFCBANK.NS": 0.00042, "INFY.NS": 0.00048, "ICICIBANK.NS": 0.00044, "LT.NS": 0.00041, "SBIN.NS": 0.00047, "ITC.NS": 0.00033, "TITAN.NS": 0.00046, "AXISBANK.NS": 0.00043}
    vol_map = {"RELIANCE.NS": 0.017, "TCS.NS": 0.013, "HDFCBANK.NS": 0.014, "INFY.NS": 0.015, "ICICIBANK.NS": 0.016, "LT.NS": 0.018, "SBIN.NS": 0.021, "ITC.NS": 0.011, "TITAN.NS": 0.017, "AXISBANK.NS": 0.019}
    start_price = {"RELIANCE.NS": 2500, "TCS.NS": 3800, "HDFCBANK.NS": 1450, "INFY.NS": 1650, "ICICIBANK.NS": 1120, "LT.NS": 3400, "SBIN.NS": 780, "ITC.NS": 440, "TITAN.NS": 3300, "AXISBANK.NS": 1080}
    drift = drift_map.get(symbol, 0.0004)
    vol = vol_map.get(symbol, 0.018)
    prices = [start_price.get(symbol, 100.0)]
    returns = rng.normal(drift, vol, len(dates))
    for r in returns[1:]:
        prices.append(prices[-1] * np.exp(r))
    close = np.array(prices)
    open_ = close * (1 + rng.normal(0, 0.004, len(close)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.006, 0.004, len(close))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.006, 0.004, len(close))))
    volume = np.clip(rng.lognormal(mean=14.2, sigma=0.35, size=len(close)), 250000, 25000000)
    df = pd.DataFrame({"symbol": symbol, "date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    df["return_1d"] = df["close"].pct_change().fillna(0)
    df["ma_10"] = df["close"].rolling(10).mean().bfill()
    df["ma_20"] = df["close"].rolling(20).mean().bfill()
    df["volatility_10"] = df["return_1d"].rolling(10).std().bfill().fillna(0)
    df["momentum_20"] = df["close"].pct_change(20).bfill().fillna(0)
    return df

def load_stock_history(symbol):
    if STOCK_DATA_PATH.exists():
        df = pd.read_csv(STOCK_DATA_PATH)
        if "symbol" in df.columns:
            sdf = df[df["symbol"] == symbol].copy()
            if not sdf.empty:
                sdf["date"] = pd.to_datetime(sdf["date"])
                return sdf
    return generate_stock_history(symbol)

def make_stock_recommendations(universe):
    if not STOCK_DATA_PATH.exists():
        return None
    hist = pd.read_csv(STOCK_DATA_PATH)
    if "symbol" not in hist.columns:
        return None
    out = []
    for sym in universe["symbol"]:
        sdf = hist[hist["symbol"] == sym].copy()
        if sdf.empty:
            continue
        latest = sdf.iloc[-1]
        recent = sdf.tail(60)
        ret_20 = recent["close"].pct_change(20).iloc[-1] if len(recent) > 20 else 0
        vol_20 = recent["return_1d"].tail(20).std()
        trend = (latest["ma_10"] - latest["ma_20"]) / max(latest["ma_20"], 1e-6)
        score = 0.28 * universe.loc[universe["symbol"] == sym, "momentum"].iloc[0] + 0.22 * universe.loc[universe["symbol"] == sym, "valuation"].iloc[0] + 0.18 * np.clip(trend * 10, -1, 1) + 0.16 * np.clip(ret_20 * 5, -1, 1) - 0.14 * np.clip(vol_20 * 20, 0, 1)
        score = float(np.clip(score, 0, 1))
        reco = "BUY" if score >= 0.68 else ("HOLD" if score >= 0.45 else "SELL")
        out.append({"symbol": sym, "sector": universe.loc[universe["symbol"] == sym, "sector"].iloc[0], "recommendation_score": round(score, 3), "recommendation": reco, "last_close": round(float(latest["close"]), 2), "momentum": round(float(universe.loc[universe["symbol"] == sym, "momentum"].iloc[0]), 3), "valuation": round(float(universe.loc[universe["symbol"] == sym, "valuation"].iloc[0]), 3)})
    rec_df = pd.DataFrame(out).sort_values(["recommendation_score", "last_close"], ascending=[False, False])
    rec_df.to_csv(RECO_PATH, index=False)
    return rec_df

@st.cache_resource
def load_loan_artifacts():
    model = joblib.load(LOAN_MODEL_PATH)
    feature_cols = json.load(open(LOAN_FEATURES_PATH))
    metrics = json.load(open(LOAN_METRICS_PATH))
    df = pd.read_csv(LOAN_SCORE_PATH) if LOAN_SCORE_PATH.exists() else pd.read_csv(LOAN_DATA_PATH)
    if "default_probability" not in df.columns and LOAN_SCORE_PATH.exists():
        X = pd.get_dummies(df.drop(columns=["default"]), drop_first=False).reindex(columns=feature_cols, fill_value=0)
        df["default_probability"] = model.predict_proba(X)[:, 1]
        df["expected_loss"] = df["default_probability"] * 0.45 * df["loan_amount"]
    return model, feature_cols, metrics, df

@st.cache_resource
def load_stock_artifacts():
    model = None #load_model(MARKET_MODEL_PATH)
    scaler = joblib.load(MARKET_SCALER_PATH)
    metrics = json.load(open(MARKET_METRICS_PATH))
    forecast_df = pd.read_csv(MARKET_FORECAST_PATH)
    universe = ensure_synthetic_stock_universe()
    return model, scaler, metrics, forecast_df, universe

def make_loan_input(row_df, feature_cols):
    row = row_df.copy()
    row = pd.get_dummies(row, columns=row.select_dtypes(include=["object"]).columns.tolist(), drop_first=False)
    return row.reindex(columns=feature_cols, fill_value=0)

def loan_interest_rate_table(base_amount, tenure_months, base_credit_score):
    loan_types = [("Personal Loan", 14.5, 1.00), ("Home Loan", 8.5, 1.35), ("Car Loan", 11.0, 1.10), ("Education Loan", 10.0, 1.20), ("Business Loan", 16.0, 1.55)]
    rows = []
    for name, rate, risk_mult in loan_types:
        adj = -1.0 if base_credit_score >= 750 else (-0.4 if base_credit_score >= 700 else (0.8 if base_credit_score < 650 else 0.0))
        annual_rate = max(rate + adj, 4.5)
        monthly_rate = annual_rate / 12 / 100
        emi = (base_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1) if monthly_rate > 0 else base_amount / tenure_months
        rows.append({"Loan Type": name, "Interest Rate %": round(annual_rate, 2), "Risk Multiplier": risk_mult, "EMI": round(emi, 2), "Total Interest": round(emi * tenure_months - base_amount, 2)})
    return pd.DataFrame(rows)

st.title("Financial Risk Intelligence Suite")
st.caption("Loan default prediction, expected loss estimation, interest-rate analysis, stock forecasting, and stock recommendations.")

with st.sidebar:
    st.header("Navigation")
    module = st.radio("Choose Module", ["Loan Default & Expected Loss", "Stock Forecasting", "Stock Recommendations"])
    st.markdown("---")
    st.markdown("Train the models first by running `loan_train.py` and `stock_train.py`.")

if module == "Loan Default & Expected Loss":
    loan_model, loan_features, loan_metrics, loan_df = load_loan_artifacts()
    st.subheader("Loan Default Prediction & Expected Loss")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(loan_df):,}")
    c2.metric("Avg Default Prob.", f"{loan_df['default_probability'].mean()*100:.2f}%")
    c3.metric("Avg Expected Loss", f"{loan_df['expected_loss'].mean():,.0f}")
    c4.metric("Default Rate", f"{loan_df['default'].mean()*100:.2f}%")

    left, right = st.columns([1.2, 0.8])
    with left:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Credit Score", "Debt-to-Income", "Default Probability", "Expected Loss"))
        fig.add_trace(go.Histogram(x=loan_df["credit_score"], nbinsx=35, marker_color="#38bdf8"), row=1, col=1)
        fig.add_trace(go.Histogram(x=loan_df["debt_to_income_ratio"], nbinsx=35, marker_color="#f59e0b"), row=1, col=2)
        fig.add_trace(go.Histogram(x=loan_df["default_probability"], nbinsx=35, marker_color="#ef4444"), row=2, col=1)
        fig.add_trace(go.Histogram(x=loan_df["expected_loss"], nbinsx=35, marker_color="#a78bfa"), row=2, col=2)
        fig.update_layout(height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.write("Model Performance")
        score = pd.DataFrame([
            {"Metric": "Accuracy", "Value": loan_metrics["accuracy"]},
            {"Metric": "Precision", "Value": loan_metrics["precision"]},
            {"Metric": "Recall", "Value": loan_metrics["recall"]},
            {"Metric": "F1", "Value": loan_metrics["f1"]},
            {"Metric": "ROC AUC", "Value": loan_metrics["roc_auc"]},
        ])
        fig = px.bar(score, x="Value", y="Metric", orientation="h", text=score["Value"].map(lambda x: f"{x:.3f}"), color="Value", color_continuous_scale=["#22c55e", "#facc15", "#ef4444"], range_x=[0, 1])
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Expected Loss = PD x LGD x EAD")
        st.markdown("Here PD comes from the model and LGD is assumed at 45%.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Loan Interest Rate Analysis")
    i1, i2, i3 = st.columns(3)
    base_amount = i1.number_input("Requested Loan Amount", 0.0, 10000000.0, 250000.0)
    tenure_months = i2.number_input("Tenure Months", 1, 360, 60)
    base_credit_score = i3.slider("Credit Score", 300, 850, 700)
    rate_df = loan_interest_rate_table(base_amount, tenure_months, base_credit_score)
    st.dataframe(rate_df, use_container_width=True)
    rate_fig = px.bar(rate_df, x="Loan Type", y="Interest Rate %", color="Interest Rate %", title="Interest Rate by Loan Type", color_continuous_scale=["#22c55e", "#facc15", "#ef4444"])
    rate_fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"))
    st.plotly_chart(rate_fig, use_container_width=True)

    st.markdown("### Loan Risk Explorer")
    filt1, filt2, filt3 = st.columns(3)
    min_score = filt1.slider("Min Credit Score", 300, 850, 300)
    max_dti = filt2.slider("Max DTI", 0.0, 1.0, 1.0)
    emp_sel = filt3.multiselect("Employment Status", sorted(loan_df["employment_status"].unique().tolist()), default=sorted(loan_df["employment_status"].unique().tolist()))
    view = loan_df[(loan_df["credit_score"] >= min_score) & (loan_df["debt_to_income_ratio"] <= max_dti) & (loan_df["employment_status"].isin(emp_sel))]
    st.dataframe(view.sort_values("default_probability", ascending=False).head(100), use_container_width=True)
    st.download_button("Download Loan Scored Data", loan_df.to_csv(index=False).encode("utf-8"), "loan_scored_data.csv", "text/csv")

    st.markdown("### Manual Loan Risk Check")
    with st.form("loan_predict"):
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age", 21, 80, 35)
            annual_income = st.number_input("Annual Income", 0.0, 10000000.0, 50000.0)
            loan_amount = st.number_input("Loan Amount", 0.0, 10000000.0, 25000.0)
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            credit_history_years = st.number_input("Credit History Years", 0.0, 50.0, 5.0)
        with b:
            debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.25)
            num_delinquencies = st.number_input("Num Delinquencies", 0, 20, 0)
            num_open_accounts = st.number_input("Num Open Accounts", 1, 20, 4)
            recent_credit_inquiries = st.number_input("Recent Credit Inquiries", 0, 20, 1)
            savings_balance = st.number_input("Savings Balance", 0.0, 10000000.0, 10000.0)
        with c:
            employment_years = st.number_input("Employment Years", 0.0, 50.0, 5.0)
            interest_rate = st.number_input("Interest Rate", 0.0, 40.0, 12.0)
            loan_tenure_months = st.number_input("Loan Tenure Months", 1, 360, 36)
            employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
            rent_or_own = st.selectbox("Rent or Own", ["Own", "Rent", "Mortgage"])
            loan_purpose = st.selectbox("Loan Purpose", ["Debt consolidation", "Home improvement", "Business", "Medical", "Education", "Other"])
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        run = st.form_submit_button("Predict Default")

    if run:
        row = pd.DataFrame([{
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "debt_to_income_ratio": debt_to_income_ratio,
            "num_delinquencies": num_delinquencies,
            "num_open_accounts": num_open_accounts,
            "recent_credit_inquiries": recent_credit_inquiries,
            "savings_balance": savings_balance,
            "employment_years": employment_years,
            "interest_rate": interest_rate,
            "loan_tenure_months": loan_tenure_months,
            "employment_status": employment_status,
            "rent_or_own": rent_or_own,
            "loan_purpose": loan_purpose,
            "region": region,
        }])
        inp = make_loan_input(row, loan_features)
        prob = float(loan_model.predict_proba(inp)[0, 1])
        el = prob * 0.45 * loan_amount
        st.metric("Default Probability", f"{prob:.2%}")
        st.metric("Expected Loss", f"{el:,.0f}")
        if prob >= 0.5:
            st.error("High Default Risk")
        elif prob >= 0.25:
            st.warning("Medium Default Risk")
        else:
            st.success("Low Default Risk")

elif module == "Stock Forecasting":
    stock_model, stock_scaler, stock_metrics, stock_forecast_df, universe = load_stock_artifacts()
    st.subheader("Stock Forecasting")
    st.markdown("Select a stock, load or generate its synthetic history, and forecast the next 30 business days.")

    stock_list = universe["symbol"].tolist()
    stock_choice = st.selectbox("Select Stock", stock_list, index=0)
    horizon = st.slider("Forecast Horizon (days)", 7, 60, 30)
    refresh = st.button("Generate / Refresh Synthetic Data for Selected Stock")

    if refresh or not STOCK_DATA_PATH.exists():
        new_hist = generate_stock_history(stock_choice)
        existing = pd.read_csv(STOCK_DATA_PATH) if STOCK_DATA_PATH.exists() else pd.DataFrame(columns=new_hist.columns)
        if not existing.empty and "symbol" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"])
            existing = existing[existing["symbol"] != stock_choice]
        combined = pd.concat([existing, new_hist], ignore_index=True)
        combined.to_csv(STOCK_DATA_PATH, index=False)

    stock_df = load_stock_history(stock_choice)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Points", f"{len(stock_df):,}")
    c2.metric("Latest Close", f"{stock_df['close'].iloc[-1]:.2f}")
    c3.metric("MAE", f"{stock_metrics['mae']:.2f}")
    c4.metric("RMSE", f"{stock_metrics['rmse']:.2f}")

    left, right = st.columns([1.18, 0.82])
    with left:
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=stock_df["date"], y=stock_df["close"], name=f"{stock_choice} Close", line=dict(color="#38bdf8", width=2)))
        hist_fig.add_trace(go.Scatter(x=stock_df["date"], y=stock_df["ma_20"], name="MA 20", line=dict(color="#f59e0b", width=2)))
        hist_fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), title=f"{stock_choice} Historical Price Trend")
        st.plotly_chart(hist_fig, use_container_width=True)

        forecast_plot = stock_forecast_df[stock_forecast_df["symbol"] == stock_choice].head(horizon).copy()
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=forecast_plot["date"], y=forecast_plot["forecast_close"], name="Forecast", line=dict(color="#22c55e", width=3)))
        forecast_fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), title=f"{stock_choice} Forecast ({horizon} days)")
        st.plotly_chart(forecast_fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.write("Forecast Metrics")
        met = pd.DataFrame([
            {"Metric": "MAE", "Value": stock_metrics["mae"]},
            {"Metric": "RMSE", "Value": stock_metrics["rmse"]},
            {"Metric": "MAPE %", "Value": stock_metrics["mape"]},
        ])
        met_fig = px.bar(met, x="Value", y="Metric", orientation="h", text=met["Value"].map(lambda x: f"{x:.3f}"), color="Value", color_continuous_scale=["#22c55e", "#facc15", "#ef4444"])
        met_fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"), showlegend=False)
        met_fig.update_traces(textposition="outside")
        st.plotly_chart(met_fig, use_container_width=True)
        st.markdown("The forecast line shows the selected stock path for the chosen horizon.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.download_button("Download Forecast", stock_forecast_df[stock_forecast_df["symbol"] == stock_choice].head(horizon).to_csv(index=False).encode("utf-8"), "stock_forecast.csv", "text/csv")

else:
    st.subheader("Stock Recommendations")
    st.markdown("Stocks are scored using synthetic fundamentals plus recent synthetic price behavior.")
    universe = ensure_synthetic_stock_universe()
    if not STOCK_DATA_PATH.exists():
        all_hist = pd.concat([generate_stock_history(sym) for sym in universe["symbol"]], ignore_index=True)
        all_hist.to_csv(STOCK_DATA_PATH, index=False)
    else:
        hist = pd.read_csv(STOCK_DATA_PATH)
        if "symbol" not in hist.columns or hist["symbol"].nunique() < 3:
            all_hist = pd.concat([generate_stock_history(sym) for sym in universe["symbol"]], ignore_index=True)
            all_hist.to_csv(STOCK_DATA_PATH, index=False)

    rec_df = make_stock_recommendations(universe)
    if rec_df is None or rec_df.empty:
        st.error("Recommendation data could not be built yet. Generate stock history first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Stocks Scored", f"{len(rec_df)}")
        c2.metric("BUY Candidates", f"{(rec_df['recommendation'] == 'BUY').sum()}")
        c3.metric("Top Score", f"{rec_df['recommendation_score'].max():.3f}")
        st.dataframe(rec_df, use_container_width=True)
        fig = px.scatter(rec_df, x="valuation", y="momentum", color="recommendation", size="recommendation_score", hover_name="symbol", title="Stock Recommendation Map")
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)", font=dict(color="#edf2f7"))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download Recommendations", rec_df.to_csv(index=False).encode("utf-8"), "stock_recommendations.csv", "text/csv")
