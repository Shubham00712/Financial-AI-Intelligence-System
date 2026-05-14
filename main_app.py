import streamlit as st
import os


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Financial AI Intelligence Suite",
    page_icon="💳",
    layout="wide"
)


# =====================================================
# ENHANCED CUSTOM CSS THEME
# =====================================================

st.markdown(
    """
    <style>
    :root {
        --bg-main: #06121f;
        --bg-soft: #0c1b2a;
        --sidebar-bg: #0a1624;
        --card-bg: linear-gradient(180deg, rgba(18, 34, 52, 0.96), rgba(11, 23, 37, 0.98));
        --card-border: rgba(84, 160, 255, 0.16);
        --text-main: #eef4fb;
        --text-soft: #9eb0c3;
        --cyan: #22d3ee;
        --blue: #4f8cff;
        --teal: #21c7b8;
        --gold: #f6c453;
        --success: #36d399;
        --danger: #ff6b6b;
        --shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(79, 140, 255, 0.16), transparent 25%),
            radial-gradient(circle at left top, rgba(33, 199, 184, 0.12), transparent 22%),
            linear-gradient(135deg, var(--bg-main) 0%, var(--bg-soft) 100%);
        color: var(--text-main);
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09131f 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1380px;
    }

    .title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #f8fbff;
        text-align: center;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }

    .subtitle {
        font-size: 1.05rem;
        color: var(--text-soft);
        text-align: center;
        margin-bottom: 1.8rem;
    }

    .card {
        background: var(--card-bg);
        color: var(--text-main);
        padding: 24px 22px;
        border-radius: 20px;
        margin-bottom: 12px;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow);
        min-height: 210px;
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.05), rgba(79, 140, 255, 0.04));
        pointer-events: none;
    }

    .feature-title {
        color: #8be9ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        line-height: 1.3;
    }

    .card-text {
        color: var(--text-soft);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .feature-box {
        background: linear-gradient(180deg, rgba(17, 36, 56, 0.94), rgba(11, 24, 38, 0.98));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        color: var(--text-soft);
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }

    .feature-box strong {
        color: #dff7ff;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(79, 140, 255, 0.16);
        background: linear-gradient(135deg, var(--blue), var(--teal));
        color: white;
        font-weight: 700;
        padding: 0.70rem 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 8px 20px rgba(33, 199, 184, 0.12);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(33, 199, 184, 0.18);
        border-color: rgba(255,255,255,0.14);
    }

    .stButton > button:focus {
        outline: none !important;
        box-shadow: 0 0 0 0.18rem rgba(34, 211, 238, 0.18);
    }

    div[data-baseweb="radio"] > div {
        gap: 0.35rem;
    }

    .stRadio > div {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 0.4rem 0.45rem;
    }

    .stMarkdown, p, label, div {
        color: var(--text-main);
    }

    h1, h2, h3 {
        color: #f7fbff;
    }

    .stAlert {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 0.75rem 1rem;
        box-shadow: var(--shadow);
    }

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0 1.3rem 0;
    }

    .status-ok {
        background: rgba(54, 211, 153, 0.12);
        color: #7ef0bf;
        padding: 0.85rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(54, 211, 153, 0.22);
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="title">💳 Financial AI Intelligence Suite</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Integrated Banking Risk • Fraud • Analytics • AML Intelligence Platform</div>',
    unsafe_allow_html=True
)


# =====================================================
# SESSION STATE + SIDEBAR NAVIGATION
# =====================================================

menu_options = [
    "🏠 Home",
    "📊 Financial Analytics",
    "🚨 Fraud Detection",
    "📉 Loan Default Prediction",
    "🛡️ Risk, AML & Sentiment Intelligence",
    "🧾 InsurTech Analytics, Integration & Final Pitch"
]

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "🏠 Home"

st.sidebar.title("📌 Navigation")

current_index = menu_options.index(st.session_state.selected_menu)

menu = st.sidebar.radio(
    "Select Module",
    menu_options,
    index=current_index
)

st.session_state.selected_menu = menu


# =====================================================
# HOME PAGE
# =====================================================

if menu == "🏠 Home":

    st.markdown("---")

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        st.markdown(
            """
            <div class="card">
                <div class="feature-title">📉 Loan Risk Intelligence</div>
                <div class="card-text">
                    Predict customer loan default risk using machine learning models and support proactive credit decisions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Loan Default Prediction", key="loan_btn", use_container_width=True):
            st.session_state.selected_menu = "📉 Loan Default Prediction"
            st.rerun()

    with row1_col2:
        st.markdown(
            """
            <div class="card">
                <div class="feature-title">🚨 Fraud Monitoring</div>
                <div class="card-text">
                    Detect suspicious financial transactions, monitor anomalies, and surface fraud patterns in real time.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Fraud Detection", key="fraud_btn", use_container_width=True):
            st.session_state.selected_menu = "🚨 Fraud Detection"
            st.rerun()

    with row1_col3:
        st.markdown(
            """
            <div class="card">
                <div class="feature-title">📊 Financial Analytics</div>
                <div class="card-text">
                    Visualize customer insights, performance trends, and financial behavior through interactive dashboards.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Financial Analytics", key="analytics_btn", use_container_width=True):
            st.session_state.selected_menu = "📊 Financial Analytics"
            st.rerun()

    with row2_col1:
        st.markdown(
            """
            <div class="card">
                <div class="feature-title">🛡️ Risk, AML & Sentiment</div>
                <div class="card-text">
                    Comply, monitor, and stay ahead of the market with integrated AML surveillance and sentiment intelligence.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Risk, AML & Sentiment", key="aml_btn", use_container_width=True):
            st.session_state.selected_menu = "🛡️ Risk, AML & Sentiment Intelligence"
            st.rerun()

    with row2_col2:
        st.markdown(
            """
            <div class="card">
                <div class="feature-title">🧾 InsurTech Analytics</div>
                <div class="card-text">
                    Price risk, integrate operations, and present insurance performance in a board-ready format.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open InsurTech Analytics", key="insurtech_btn", use_container_width=True):
            st.session_state.selected_menu = "🧾 InsurTech Analytics, Integration & Final Pitch"
            st.rerun()

    st.markdown("---")

    st.subheader("📌 Platform Features")

    st.markdown("""
    <div class="feature-box"><strong>AI-Powered Loan Default Prediction</strong><br>Predict borrower-level risk and support better lending decisions.</div>
    <div class="feature-box"><strong>Real-Time Fraud Detection</strong><br>Monitor suspicious activity and identify high-risk transactions quickly.</div>
    <div class="feature-box"><strong>Banking Analytics Dashboard</strong><br>Track trends, customer patterns, and key financial indicators.</div>
    <div class="feature-box"><strong>AML & Sentiment Intelligence</strong><br>Combine compliance monitoring with market and reputation signals.</div>
    <div class="feature-box"><strong>InsurTech Pricing and Claims Analytics</strong><br>Evaluate pricing, claims behavior, and operational integration metrics.</div>
    <div class="feature-box"><strong>Executive-Level Decision Support</strong><br>Turn advanced analytics into business-ready financial intelligence.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="status-ok">System Ready ✔️</div>', unsafe_allow_html=True)


# =====================================================
# FINANCIAL ANALYTICS MODULE
# =====================================================

elif menu == "📊 Financial Analytics":

    st.title("📊 Financial Analytics Dashboard")
    st.info("Launching Financial Analytics Project...")

    project_path = r"Financial EDA & Credit Scoring/app.py"

    if os.path.exists(project_path):
        with open(project_path, encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        st.error("Financial Analytics app.py file not found")


# =====================================================
# FRAUD DETECTION MODULE
# =====================================================

elif menu == "🚨 Fraud Detection":

    st.title("🚨 Fraud Detection System")
    st.info("Launching Fraud Detection Project...")

    project_path = r"Fraud_Detection/app.py"

    if os.path.exists(project_path):
        with open(project_path, encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        st.error("Fraud Detection app.py file not found")


# =====================================================
# LOAN DEFAULT MODULE
# =====================================================

elif menu == "📉 Loan Default Prediction":

    st.title("📉 Loan Default Prediction System")
    st.info("Launching Loan Default Project...")

    project_path = r"Loan_Default_Pipeline_Market_Forecasting/app.py"

    if os.path.exists(project_path):
        with open(project_path, encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        st.error("Loan Default app.py file not found")


# =====================================================
# RISK, AML & SENTIMENT MODULE
# =====================================================

elif menu == "🛡️ Risk, AML & Sentiment Intelligence":

    st.title("🛡️ Risk Dashboard, AML & Sentiment Intelligence")
    st.info("Launching Risk, AML & Sentiment Intelligence Project...")

    project_path = r"./risk_dashboard_aml_sentiment_app.py"

    if os.path.exists(project_path):
        with open(project_path, encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        st.error("Risk_AML_Sentiment app.py file not found")


elif menu == "🧾 InsurTech Analytics, Integration & Final Pitch":

    st.title("🧾 InsurTech Analytics, Integration & Final Pitch")
    st.info("Launching InsurTech Analytics Project...")

    project_path = r"./insurtech_analytics_app.py"

    if os.path.exists(project_path):
        with open(project_path, encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        st.error("InsurTech_Analytics app.py file not found")