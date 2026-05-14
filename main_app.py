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
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    .stSidebar {
        background-color: #111827;
    }

    .title {
        font-size: 42px;
        font-weight: bold;
        color: #00E5FF;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 40px;
    }

    .card {
        background-color: #1F2937;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
        text-align: center;
        min-height: 180px;
    }

    .feature-title {
        color: #00E5FF;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
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
                <br>
                Predict customer loan default risk using machine learning models.
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
                <br>
                Detect suspicious financial transactions and fraudulent activity.
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
                <br>
                Visualize customer insights, trends, and banking performance.
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
                <br>
                Comply, monitor, and stay ahead of the market with integrated risk intelligence.
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
                <br>
                Price risk, integrate operations, and present insurance performance to the board.
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open InsurTech Analytics", key="insurtech_btn", use_container_width=True):
            st.session_state.selected_menu = "🧾 InsurTech Analytics, Integration & Final Pitch"
            st.rerun()

    # with row2_col3:
        

    st.markdown("---")

    st.subheader("📌 Platform Features")

    st.write("""
    ✅ AI-Powered Loan Default Prediction

    ✅ Real-Time Fraud Detection

    ✅ Banking Analytics Dashboard

    ✅ AML & Sentiment Intelligence

    ✅ InsurTech Pricing, Claims, and Integration Dashboard

    ✅ Executive-Level Financial Decision Support
    """)

    st.markdown("---")
    st.success("System Ready ✔️")

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