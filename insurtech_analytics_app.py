import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# st.set_page_config(
#     page_title="InsurTech Analytics, Integration & Final Pitch",
#     page_icon="🧾",
#     layout="wide"
# )

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        color: #e5eefc;
    }
    .stSidebar {
        background: #111827;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #0f766e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #dbeafe;
        font-size: 1rem;
    }
    .section-card {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem;
        border-radius: 18px;
        margin-bottom: 1rem;
    }
    .mini-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.12);
        color: #93c5fd;
        font-size: 0.82rem;
        margin-right: 0.4rem;
    }
    .good-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(34,197,94,0.12);
        color: #86efac;
        font-size: 0.82rem;
        margin-right: 0.4rem;
    }
    .warn-tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(245,158,11,0.12);
        color: #fcd34d;
        font-size: 0.82rem;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🧾 InsurTech Analytics, Integration & Final Pitch</div>
    <div class="hero-subtitle">Price Risk, Integrate, and Present to the Board</div>
</div>
""", unsafe_allow_html=True)

np.random.seed(21)

@st.cache_data
def generate_insurance_data():
    months = pd.date_range("2024-01-01", periods=18, freq="MS")
    lob = ["Motor", "Health", "Life", "Travel", "Property", "SME"]
    channels = ["Direct", "Bancassurance", "Broker", "Digital", "Agent"]
    regions = ["North", "South", "East", "West", "Central"]

    rows = []
    for m in months:
        for l in lob:
            for c in channels:
                premium = np.random.randint(180000, 950000)
                claims = np.random.randint(80000, 720000)
                expenses = np.random.randint(30000, 180000)
                policies = np.random.randint(350, 2200)
                renewals = int(policies * np.random.uniform(0.62, 0.91))
                claim_count = np.random.randint(40, 550)
                avg_claim = claims / max(claim_count, 1)
                rows.append([
                    m, l, c,
                    np.random.choice(regions),
                    premium, claims, expenses, policies,
                    renewals, claim_count, avg_claim
                ])

    df = pd.DataFrame(rows, columns=[
        "month", "line_of_business", "channel", "region",
        "gross_written_premium", "incurred_claims", "operating_expense",
        "policies", "renewals", "claim_count", "avg_claim_amount"
    ])

    df["loss_ratio"] = (df["incurred_claims"] / df["gross_written_premium"] * 100).round(2)
    df["expense_ratio"] = (df["operating_expense"] / df["gross_written_premium"] * 100).round(2)
    df["combined_ratio"] = (df["loss_ratio"] + df["expense_ratio"]).round(2)
    df["retention_rate"] = (df["renewals"] / df["policies"] * 100).round(2)
    df["underwriting_margin"] = (
        df["gross_written_premium"] - df["incurred_claims"] - df["operating_expense"]
    )
    df["claim_frequency"] = (df["claim_count"] / df["policies"] * 100).round(2)

    quote_n = 300
    quotes = pd.DataFrame({
        "quote_id": [f"QT-{10000+i}" for i in range(quote_n)],
        "product": np.random.choice(lob, quote_n),
        "channel": np.random.choice(channels, quote_n),
        "risk_score": np.round(np.random.uniform(0.1, 0.98, quote_n), 2),
        "proposed_premium": np.random.randint(8000, 120000, quote_n),
        "conversion_prob": np.round(np.random.uniform(0.25, 0.95, quote_n), 2),
        "status": np.random.choice(["Priced", "Pending", "Bound", "Declined"], quote_n, p=[0.28, 0.27, 0.32, 0.13]),
        "integration_status": np.random.choice(["Integrated", "Pending API", "Manual"], quote_n, p=[0.63, 0.21, 0.16])
    })

    board_initiatives = pd.DataFrame({
        "initiative": [
            "Pricing Model Refresh",
            "Claims Automation",
            "Digital Quote-to-Bind",
            "Fraud & Leakage Controls",
            "Partner API Expansion",
            "Board Reporting Pack"
        ],
        "owner": ["Pricing", "Claims Ops", "Distribution", "Risk", "Tech", "Strategy"],
        "status": ["On Track", "At Risk", "On Track", "On Track", "At Risk", "On Track"],
        "progress_pct": [72, 58, 76, 68, 54, 81]
    })

    return df, quotes, board_initiatives

df, quotes, board_initiatives = generate_insurance_data()

st.sidebar.header("⚙️ Filters")
lob_filter = st.sidebar.multiselect(
    "Line of Business",
    sorted(df["line_of_business"].unique()),
    default=sorted(df["line_of_business"].unique())
)
channel_filter = st.sidebar.multiselect(
    "Distribution Channel",
    sorted(df["channel"].unique()),
    default=sorted(df["channel"].unique())
)
region_filter = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

filtered = df[
    df["line_of_business"].isin(lob_filter) &
    df["channel"].isin(channel_filter) &
    df["region"].isin(region_filter)
].copy()

total_gwp = filtered["gross_written_premium"].sum()
total_claims = filtered["incurred_claims"].sum()
total_expense = filtered["operating_expense"].sum()
avg_loss_ratio = round((total_claims / total_gwp) * 100, 2) if total_gwp else 0
avg_expense_ratio = round((total_expense / total_gwp) * 100, 2) if total_gwp else 0
combined_ratio = round(avg_loss_ratio + avg_expense_ratio, 2)
retention_rate = round(filtered["renewals"].sum() / filtered["policies"].sum() * 100, 2) if filtered["policies"].sum() else 0
uw_margin = filtered["underwriting_margin"].sum()
avg_claim = round(filtered["avg_claim_amount"].mean(), 2)
claim_frequency = round(filtered["claim_frequency"].mean(), 2)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Gross Written Premium", f"₹{total_gwp/1e6:,.1f}M")
k2.metric("Loss Ratio", f"{avg_loss_ratio}%")
k3.metric("Combined Ratio", f"{combined_ratio}%")
k4.metric("Retention Rate", f"{retention_rate}%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Underwriting Margin", f"₹{uw_margin/1e6:,.1f}M")
k6.metric("Avg Claim Amount", f"₹{avg_claim:,.0f}")
k7.metric("Claim Frequency", f"{claim_frequency}%")
k8.metric("Integrated Quote Flows", f"{(quotes['integration_status']=='Integrated').sum()}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Overview",
    "Pricing & Risk",
    "Claims & Operations",
    "Distribution & Integration",
    "Board Pack",
    "Pitch Summary"
])

with tab1:
    c1, c2 = st.columns([1.3, 1])

    monthly = filtered.groupby("month", as_index=False).agg({
        "gross_written_premium": "sum",
        "incurred_claims": "sum",
        "operating_expense": "sum",
        "underwriting_margin": "sum"
    })
    monthly["loss_ratio"] = (monthly["incurred_claims"] / monthly["gross_written_premium"] * 100).round(2)
    monthly["combined_ratio"] = (
        (monthly["incurred_claims"] + monthly["operating_expense"]) / monthly["gross_written_premium"] * 100
    ).round(2)

    with c1:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["gross_written_premium"],
            mode="lines+markers", name="GWP", line=dict(color="#38bdf8", width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["underwriting_margin"],
            mode="lines+markers", name="UW Margin", line=dict(color="#22c55e", width=3)
        ))
        fig_trend.update_layout(
            title="Premium and Underwriting Trend",
            template="plotly_dark",
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        ratio_df = pd.DataFrame({
            "Metric": ["Loss Ratio", "Expense Ratio", "Combined Ratio"],
            "Value": [avg_loss_ratio, avg_expense_ratio, combined_ratio]
        })
        fig_ratio = px.bar(
            ratio_df, x="Metric", y="Value", color="Metric",
            title="Core Insurance Ratios",
            template="plotly_dark",
            color_discrete_sequence=["#f59e0b", "#60a5fa", "#ef4444"]
        )
        fig_ratio.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Management View")
    st.write(
        "This dashboard brings pricing discipline, underwriting profitability, claims efficiency, partner integration, and executive storytelling into one InsurTech control layer so teams can price smarter, integrate faster, and present clearly to leadership."
    )
    st.markdown(
        '<span class="mini-tag">Pricing intelligence</span>'
        '<span class="mini-tag">Claims visibility</span>'
        '<span class="good-tag">Board-ready narrative</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    p1, p2 = st.columns(2)

    lob_perf = filtered.groupby("line_of_business", as_index=False).agg({
        "gross_written_premium": "sum",
        "incurred_claims": "sum",
        "operating_expense": "sum",
        "underwriting_margin": "sum"
    })
    lob_perf["loss_ratio"] = (lob_perf["incurred_claims"] / lob_perf["gross_written_premium"] * 100).round(2)
    lob_perf["combined_ratio"] = (
        (lob_perf["incurred_claims"] + lob_perf["operating_expense"]) / lob_perf["gross_written_premium"] * 100
    ).round(2)

    with p1:
        fig_lob = px.bar(
            lob_perf.sort_values("gross_written_premium", ascending=False),
            x="line_of_business",
            y=["gross_written_premium", "underwriting_margin"],
            barmode="group",
            title="Line of Business Performance",
            template="plotly_dark"
        )
        fig_lob.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_lob, use_container_width=True)

    with p2:
        fig_lob_ratio = px.scatter(
            lob_perf,
            x="loss_ratio",
            y="combined_ratio",
            size="gross_written_premium",
            color="underwriting_margin",
            hover_name="line_of_business",
            title="Pricing Risk Position by Line",
            template="plotly_dark",
            color_continuous_scale="RdYlGn"
        )
        fig_lob_ratio.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_lob_ratio, use_container_width=True)

    st.subheader("Quoted Risk Portfolio")
    st.dataframe(
        quotes.sort_values(["risk_score", "conversion_prob"], ascending=[False, False]).head(20),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    o1, o2 = st.columns(2)

    claims_monthly = filtered.groupby("month", as_index=False).agg({
        "claim_count": "sum",
        "incurred_claims": "sum",
        "avg_claim_amount": "mean"
    })

    with o1:
        fig_claims = px.line(
            claims_monthly,
            x="month",
            y=["claim_count", "incurred_claims"],
            title="Claims Volume and Cost Trend",
            template="plotly_dark"
        )
        fig_claims.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_claims, use_container_width=True)

    with o2:
        region_claims = filtered.groupby("region", as_index=False).agg({
            "claim_count": "sum",
            "avg_claim_amount": "mean"
        })
        fig_region = px.bar(
            region_claims,
            x="region",
            y=["claim_count", "avg_claim_amount"],
            barmode="group",
            title="Regional Claims View",
            template="plotly_dark"
        )
        fig_region.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("Operational Signals")
    op_signals = pd.DataFrame({
        "Signal": [
            "High severity in Property",
            "Margin compression in Health",
            "Improving retention in Motor",
            "Broker productivity variance",
            "Digital quote conversion uplift"
        ],
        "Impact": ["High", "High", "Medium", "Medium", "High"],
        "Recommended Action": [
            "Tighten pricing guardrails",
            "Reprice high-risk segments",
            "Protect profitable cohorts",
            "Review incentives and mix",
            "Scale digital acquisition"
        ]
    })
    st.dataframe(op_signals, use_container_width=True, hide_index=True)

with tab4:
    d1, d2 = st.columns([1.15, 1])

    with d1:
        channel_perf = filtered.groupby("channel", as_index=False).agg({
            "gross_written_premium": "sum",
            "policies": "sum",
            "renewals": "sum",
            "underwriting_margin": "sum"
        })
        channel_perf["retention_rate"] = (channel_perf["renewals"] / channel_perf["policies"] * 100).round(2)

        fig_channel = px.bar(
            channel_perf.sort_values("gross_written_premium", ascending=False),
            x="channel",
            y=["gross_written_premium", "underwriting_margin"],
            barmode="group",
            title="Channel Revenue and Margin",
            template="plotly_dark"
        )
        fig_channel.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_channel, use_container_width=True)

    with d2:
        integration_status = quotes["integration_status"].value_counts().reset_index()
        integration_status.columns = ["integration_status", "count"]
        fig_int = px.pie(
            integration_status,
            names="integration_status",
            values="count",
            hole=0.5,
            title="Integration Readiness",
            template="plotly_dark",
            color="integration_status",
            color_discrete_map={
                "Integrated": "#22c55e",
                "Pending API": "#f59e0b",
                "Manual": "#ef4444"
            }
        )
        fig_int.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_int, use_container_width=True)

    st.subheader("Integration Queue")
    st.dataframe(
        quotes[["quote_id", "product", "channel", "risk_score", "proposed_premium", "status", "integration_status"]]
        .sort_values(["integration_status", "risk_score"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True
    )

with tab5:
    b1, b2 = st.columns([1.15, 1])

    with b1:
        st.subheader("Board Initiative Tracker")
        st.dataframe(board_initiatives, use_container_width=True, hide_index=True)

    with b2:
        board_kpis = pd.DataFrame({
            "Board KPI": [
                "Combined Ratio",
                "Retention Rate",
                "Underwriting Margin",
                "Digital Integration Coverage",
                "High-Risk Quote Share"
            ],
            "Current": [
                f"{combined_ratio}%",
                f"{retention_rate}%",
                f"₹{uw_margin/1e6:,.1f}M",
                f"{round((quotes['integration_status']=='Integrated').mean()*100,1)}%",
                f"{round((quotes['risk_score']>0.75).mean()*100,1)}%"
            ],
            "Board Signal": [
                "Watch if >100%",
                "Protect core book",
                "Scale profitable lines",
                "Accelerate partner readiness",
                "Refine pricing controls"
            ]
        })
        st.dataframe(board_kpis, use_container_width=True, hide_index=True)

    st.subheader("Board Narrative")
    st.write("""
    1. Premium growth is strongest where pricing remains disciplined and distribution is integrated.
    2. Combined ratio pressure is concentrated in selected lines, making repricing and claim controls the immediate lever.
    3. Digital and partner integrations are improving conversion and operating efficiency, but manual flows still delay scale.
    4. The board story is clear: improve pricing precision, raise integration coverage, and defend underwriting margin.
    """)

with tab6:
    st.subheader("Final Pitch")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    ### Price Risk, Integrate, and Present to the Board

    **Why this matters**
    - Insurance profitability depends on pricing accuracy, claims control, retention strength, and operational efficiency.
    - A unified InsurTech dashboard helps leadership see which lines grow profitably, which channels scale efficiently, and where integration gaps are still dragging performance.

    **What this application delivers**
    - Executive view of premium, claims, retention, and underwriting margin
    - Pricing risk view by product and quote quality
    - Claims and operations visibility for severity, frequency, and regional pressure
    - Distribution and integration dashboard for digital, broker, and partner performance
    - Board-ready summary to support strategic decision making

    **Recommended next steps**
    - Reprice high combined-ratio segments
    - Prioritize API integration for manual quote journeys
    - Tighten underwriting controls on high-risk quotes
    - Expand channels with strong margin and retention
    """)
    st.markdown(
        '<span class="mini-tag">Board-ready</span>'
        '<span class="mini-tag">Pricing discipline</span>'
        '<span class="warn-tag">Integration acceleration</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")