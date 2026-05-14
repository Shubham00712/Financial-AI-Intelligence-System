import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# st.set_page_config(
#     page_title="Risk Dashboard, AML & Sentiment Intelligence",
#     page_icon="🛡️",
#     layout="wide"
# )

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: #e5eefc;
    }
    .stSidebar {
        background: #111827;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #111827 0%, #0f3d5e 45%, #134e4a 100%);
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
        color: #cbd5e1;
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
    .risk-high {
        color: #f87171;
        font-weight: 700;
    }
    .risk-med {
        color: #fbbf24;
        font-weight: 700;
    }
    .risk-low {
        color: #4ade80;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ Risk Dashboard, AML & Sentiment Intelligence</div>
    <div class="hero-subtitle">Comply, Monitor, and Stay Ahead of the Market</div>
</div>
""", unsafe_allow_html=True)

np.random.seed(42)

@st.cache_data
def generate_data():
    n_entities = 2000
    entities = [f"ENT-{1000+i}" for i in range(n_entities)]
    names = [f"Client {i}" for i in range(1, n_entities + 1)]
    countries = np.random.choice(
        ["India", "UAE", "Singapore", "UK", "USA", "Hong Kong", "Nigeria", "Germany"],
        size=n_entities,
        p=[0.28, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.14]
    )
    sectors = np.random.choice(
        ["Retail Banking", "Corporate", "MSME", "Crypto-linked", "NBFC", "Payments", "Trade Finance"],
        size=n_entities
    )
    pep = np.random.choice([0, 1], size=n_entities, p=[0.9, 0.1])
    sanctions_hit = np.random.choice([0, 1], size=n_entities, p=[0.94, 0.06])
    adverse_media = np.random.choice([0, 1], size=n_entities, p=[0.72, 0.28])
    kyc_overdue = np.random.choice([0, 1], size=n_entities, p=[0.82, 0.18])
    txn_velocity = np.random.gamma(2.3, 18, size=n_entities)
    suspicious_flags = np.random.poisson(1.5, size=n_entities)
    sentiment = np.clip(np.random.normal(0.15, 0.55, size=n_entities), -1, 1)

    risk_score = (
        pep * 20 +
        sanctions_hit * 32 +
        adverse_media * 16 +
        kyc_overdue * 10 +
        suspicious_flags * 5 +
        np.where(sectors == "Crypto-linked", 14, 0) +
        np.where(countries == "Nigeria", 10, 0) +
        np.where(countries == "UAE", 5, 0) +
        np.clip(txn_velocity / 8, 0, 18) +
        np.where(sentiment < -0.4, 12, 0)
    )

    risk_score = np.clip(risk_score + np.random.normal(0, 4, n_entities), 5, 100).round(1)
    risk_band = pd.cut(
        risk_score,
        bins=[0, 35, 65, 100],
        labels=["Low", "Medium", "High"]
    )

    df_entities = pd.DataFrame({
        "entity_id": entities,
        "entity_name": names,
        "country": countries,
        "sector": sectors,
        "pep": pep,
        "sanctions_hit": sanctions_hit,
        "adverse_media": adverse_media,
        "kyc_overdue": kyc_overdue,
        "txn_velocity": np.round(txn_velocity, 2),
        "suspicious_flags": suspicious_flags,
        "sentiment_score": np.round(sentiment, 2),
        "risk_score": risk_score,
        "risk_band": risk_band.astype(str)
    })

    start_date = datetime.today() - timedelta(days=179)
    dates = pd.date_range(start_date, periods=180, freq="D")

    daily_alerts = pd.DataFrame({
        "date": dates,
        "alerts": np.random.poisson(42, len(dates)),
        "cases_opened": np.random.poisson(11, len(dates)),
        "cases_closed": np.random.poisson(10, len(dates)),
        "sars_filed": np.random.poisson(3, len(dates)),
        "negative_news": np.random.poisson(6, len(dates)),
        "market_stress_index": np.clip(np.random.normal(54, 14, len(dates)), 10, 95)
    })

    daily_alerts["alert_to_case_rate"] = np.round(
        daily_alerts["cases_opened"] / daily_alerts["alerts"].replace(0, np.nan) * 100, 1
    ).fillna(0)

    watchlist = df_entities[df_entities["sanctions_hit"] == 1].copy().head(20)
    watchlist["watchlist_type"] = np.random.choice(
        ["Sanctions", "PEP", "Adverse Media", "Internal Blacklist"],
        len(watchlist)
    )
    watchlist["last_reviewed"] = pd.to_datetime("today") - pd.to_timedelta(
        np.random.randint(1, 45, len(watchlist)), unit="D"
    )

    case_queue = df_entities.sort_values("risk_score", ascending=False).head(40).copy()
    case_queue["case_id"] = [f"CASE-{5000+i}" for i in range(len(case_queue))]
    case_queue["priority"] = pd.cut(
        case_queue["risk_score"], bins=[0, 45, 70, 100], labels=["P3", "P2", "P1"]
    ).astype(str)
    case_queue["status"] = np.random.choice(
        ["Open", "Under Review", "Escalated", "Closed"],
        size=len(case_queue),
        p=[0.35, 0.3, 0.2, 0.15]
    )
    case_queue["owner"] = np.random.choice(
        ["Analyst A", "Analyst B", "Analyst C", "Lead AML", "Risk Ops"],
        size=len(case_queue)
    )

    return df_entities, daily_alerts, watchlist, case_queue

df_entities, daily_alerts, watchlist, case_queue = generate_data()

st.sidebar.header("⚙️ Controls")
country_filter = st.sidebar.multiselect(
    "Country", options=sorted(df_entities["country"].unique()),
    default=sorted(df_entities["country"].unique())
)
sector_filter = st.sidebar.multiselect(
    "Sector", options=sorted(df_entities["sector"].unique()),
    default=sorted(df_entities["sector"].unique())
)
risk_filter = st.sidebar.multiselect(
    "Risk Band", options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)

filtered = df_entities[
    df_entities["country"].isin(country_filter) &
    df_entities["sector"].isin(sector_filter) &
    df_entities["risk_band"].isin(risk_filter)
].copy()

total_entities = len(filtered)
high_risk = int((filtered["risk_band"] == "High").sum())
open_cases = int((case_queue["status"].isin(["Open", "Under Review", "Escalated"])).sum())
watch_hits = int(filtered["sanctions_hit"].sum())
neg_sent_share = round((filtered["sentiment_score"] < 0).mean() * 100, 1) if total_entities else 0
avg_risk = round(filtered["risk_score"].mean(), 1) if total_entities else 0
kyc_overdue = int(filtered["kyc_overdue"].sum())
sar_rate = round(daily_alerts["sars_filed"].sum() / daily_alerts["alerts"].sum() * 100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entities Monitored", f"{total_entities}")
c2.metric("High-Risk Exposure", f"{high_risk}")
c3.metric("Open AML Cases", f"{open_cases}")
c4.metric("Watchlist Hits", f"{watch_hits}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg Risk Score", f"{avg_risk}")
c6.metric("Negative Sentiment Share", f"{neg_sent_share}%")
c7.metric("KYC Reviews Overdue", f"{kyc_overdue}")
c8.metric("SAR Filing Rate", f"{sar_rate}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "AML Monitoring",
    "Sentiment Intelligence",
    "Risk Segmentation",
    "Case & Watchlist Desk"
])

with tab1:
    left, right = st.columns([1.25, 1])

    with left:
        fig_alerts = go.Figure()
        fig_alerts.add_trace(go.Scatter(
            x=daily_alerts["date"], y=daily_alerts["alerts"],
            mode="lines", name="Alerts", line=dict(color="#38bdf8", width=2.5)
        ))
        fig_alerts.add_trace(go.Scatter(
            x=daily_alerts["date"], y=daily_alerts["cases_opened"],
            mode="lines", name="Cases Opened", line=dict(color="#f59e0b", width=2)
        ))
        fig_alerts.add_trace(go.Scatter(
            x=daily_alerts["date"], y=daily_alerts["cases_closed"],
            mode="lines", name="Cases Closed", line=dict(color="#22c55e", width=2)
        ))
        fig_alerts.update_layout(
            title="Alert and Case Flow",
            template="plotly_dark",
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_alerts, use_container_width=True)

    with right:
        risk_dist = filtered["risk_band"].value_counts().reset_index()
        risk_dist.columns = ["risk_band", "count"]
        fig_pie = px.pie(
            risk_dist, names="risk_band", values="count",
            color="risk_band",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
            title="Portfolio Risk Distribution",
            hole=0.48
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Executive Summary")
    st.write(
        "This integrated dashboard combines portfolio risk monitoring, AML operations, adverse media sentiment, and market context into one control tower so teams can prioritize high-risk entities, monitor alert pressure, and respond faster to compliance-sensitive changes."
    )
    st.markdown(
        '<span class="mini-tag">Outcome-focused KPIs</span>'
        '<span class="mini-tag">Continuous monitoring</span>'
        '<span class="warn-tag">Investigation prioritization</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    a1, a2 = st.columns(2)

    with a1:
        aml_country = filtered.groupby("country", as_index=False).agg(
            high_risk_entities=("risk_band", lambda x: (x == "High").sum()),
            watchlist_hits=("sanctions_hit", "sum"),
            overdue_kyc=("kyc_overdue", "sum")
        )
        fig_country = px.bar(
            aml_country.sort_values("high_risk_entities", ascending=False),
            x="country", y=["high_risk_entities", "watchlist_hits", "overdue_kyc"],
            barmode="group",
            title="Country-Level AML Pressure",
            template="plotly_dark"
        )
        fig_country.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_country, use_container_width=True)

    with a2:
        fig_sar = px.line(
            daily_alerts,
            x="date",
            y=["alert_to_case_rate", "sars_filed"],
            title="Alert Conversion and SAR Activity",
            template="plotly_dark"
        )
        fig_sar.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sar, use_container_width=True)

    st.subheader("Top AML Risk Drivers")
    top_drivers = filtered.assign(
        driver=np.select(
            [
                filtered["sanctions_hit"] == 1,
                filtered["pep"] == 1,
                filtered["adverse_media"] == 1,
                filtered["kyc_overdue"] == 1,
                filtered["sector"] == "Crypto-linked"
            ],
            [
                "Sanctions Match",
                "PEP Exposure",
                "Adverse Media",
                "KYC Overdue",
                "Crypto-linked Business"
            ],
            default="Behavioral / Transactional"
        )
    )
    driver_tbl = top_drivers["driver"].value_counts().reset_index()
    driver_tbl.columns = ["Risk Driver", "Count"]
    st.dataframe(driver_tbl, use_container_width=True, hide_index=True)

with tab3:
    s1, s2 = st.columns([1.2, 1])

    with s1:
        sentiment_bucket = pd.cut(
            filtered["sentiment_score"],
            bins=[-1.01, -0.2, 0.2, 1.01],
            labels=["Negative", "Neutral", "Positive"]
        ).astype(str)
        sentiment_tbl = sentiment_bucket.value_counts().reset_index()
        sentiment_tbl.columns = ["Sentiment", "Count"]
        fig_sent = px.bar(
            sentiment_tbl,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={
                "Negative": "#ef4444",
                "Neutral": "#94a3b8",
                "Positive": "#22c55e"
            },
            title="Adverse Media and News Sentiment Mix",
            template="plotly_dark"
        )
        fig_sent.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with s2:
        fig_market = px.line(
            daily_alerts,
            x="date",
            y=["negative_news", "market_stress_index"],
            title="News Pressure vs Market Stress",
            template="plotly_dark"
        )
        fig_market.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_market, use_container_width=True)

    st.subheader("Entities with Negative Sentiment and Elevated Risk")
    neg_entities = filtered[(filtered["sentiment_score"] < -0.2)].sort_values("risk_score", ascending=False)[
        ["entity_id", "entity_name", "country", "sector", "sentiment_score", "risk_score", "risk_band"]
    ].head(15)
    st.dataframe(neg_entities, use_container_width=True, hide_index=True)

with tab4:
    r1, r2 = st.columns(2)

    with r1:
        sector_risk = filtered.groupby("sector", as_index=False).agg(
            avg_risk_score=("risk_score", "mean"),
            entity_count=("entity_id", "count")
        )
        fig_sector = px.scatter(
            sector_risk,
            x="entity_count",
            y="avg_risk_score",
            size="entity_count",
            color="avg_risk_score",
            hover_name="sector",
            title="Sector Risk Concentration",
            template="plotly_dark",
            color_continuous_scale="Reds"
        )
        fig_sector.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with r2:
        fig_hist = px.histogram(
            filtered,
            x="risk_score",
            nbins=25,
            color="risk_band",
            color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"},
            title="Risk Score Distribution",
            template="plotly_dark"
        )
        fig_hist.update_layout(
            height=390,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Highest-Risk Portfolio Slice")
    st.dataframe(
        filtered.sort_values("risk_score", ascending=False)[
            ["entity_id", "entity_name", "country", "sector", "risk_score", "risk_band", "pep", "sanctions_hit", "adverse_media"]
        ].head(20),
        use_container_width=True,
        hide_index=True
    )

with tab5:
    q1, q2 = st.columns([1.25, 1])

    with q1:
        st.subheader("Investigation Queue")
        st.dataframe(
            case_queue[[
                "case_id", "entity_name", "country", "sector", "risk_score",
                "priority", "status", "owner"
            ]].sort_values(["priority", "risk_score"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )

    with q2:
        st.subheader("Watchlist and Screening Review")
        st.dataframe(
            watchlist[[
                "entity_id", "entity_name", "country", "risk_score",
                "watchlist_type", "last_reviewed"
            ]],
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")