import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from pathlib import Path
import os

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Scotland Birth Rate Forecasts",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- THEME ----------
PRIMARY_COLOR = "#0d7377"
SECONDARY_COLOR = "#14a085"
BG_SIDEBAR = "#f8f9fa"
BG_MAIN = "#ffffff"
FONT = "Source Sans Pro"

st.markdown(
    f"""
    <style>
    .main {{ background-color: {BG_MAIN}; }}
    .css-1d391kg {{ background-color: {BG_SIDEBAR}; }}
    .css-1v3fvcr {{ font-family: {FONT}; }}
    .metric-card {{
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    h1, h2, h3, h4 {{ color: {PRIMARY_COLOR}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DATA ----------
@st.cache_data
def load_data():
    forecasts = pd.read_csv(os.path.join(os.path.dirname(__file__), "forecasts.csv"))

    actuals = pd.read_csv(os.path.join(os.path.dirname(__file__), "actuals.csv"))
    # Standardise date cols
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])
    actuals["Registration Year"] = pd.to_datetime(
        actuals["Registration Year"], format="%Y"
    )
    return forecasts, actuals


df_forecasts, df_actuals = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

nhs_boards = sorted(df_actuals["nhs_board"].unique())
age_groups = sorted(df_actuals["age_group"].unique())

selected_nhs_board = st.sidebar.selectbox("NHS Board", nhs_boards)
selected_age_group = st.sidebar.selectbox("Maternal Age Group", age_groups)

# ---------- FILTER ----------
mask = (
    (df_forecasts["nhs_board"] == selected_nhs_board)
    & (df_forecasts["age_group"] == selected_age_group)
)
df_forecasts_filtered = df_forecasts[mask]

mask_actual = (
    (df_actuals["nhs_board"] == selected_nhs_board)
    & (df_actuals["age_group"] == selected_age_group)
)
df_actuals_filtered = df_actuals[mask_actual]

# ---------- HEADER ----------
st.title("👶 Scotland Birth Rate Forecasts")
st.markdown(
    "Interactive dashboard to explore forecasted birth rates for NHS boards and maternal age groups in Scotland."
)

st.markdown("---")

# ---------- TIMESERIES ----------
st.subheader(f"📈 Birth Rate Trend")
with st.container():
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_actuals_filtered["Registration Year"],
            y=df_actuals_filtered["birth_rate"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=SECONDARY_COLOR, width=3, dash="dash"),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat_upper"],
            fill="tonexty",
            fillcolor="rgba(20,160,133,0.15)",
            line=dict(width=0),
            name="Uncertainty",
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Birth Rate (per 1,000 women)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- HEATMAP ----------
st.subheader("🔥 Birth-Rate Heatmap")
with st.container():
    df_heatmap = (
        df_actuals_filtered.assign(year=lambda d: d["Registration Year"].dt.year)
        .assign(month=lambda d: d["Registration Year"].dt.month)
        .pivot_table(index="month", columns="year", values="birth_rate")
    )

    fig_heatmap = px.imshow(
        df_heatmap,
        labels=dict(x="Year", y="Month", color="Birth Rate"),
        aspect="auto",
        color_continuous_scale="Teal",
    )
    fig_heatmap.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------- METRICS ----------
st.subheader("📊 Key Metrics")
latest_actual = df_actuals_filtered["birth_rate"].iloc[-1]
latest_forecast = df_forecasts_filtered["yhat"].iloc[-1]
yoy_change = ((latest_forecast - latest_actual) / latest_actual) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color:{PRIMARY_COLOR};">Latest Actual</h4>
            <h2>{latest_actual:.2f}</h2>
            <small>per 1,000 women</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color:{PRIMARY_COLOR};">Latest Forecast</h4>
            <h2>{latest_forecast:.2f}</h2>
            <small>per 1,000 women</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    color = "red" if yoy_change < 0 else "green"
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color:{PRIMARY_COLOR};">YoY Change</h4>
            <h2 style="color:{color};">{yoy_change:+.2f}%</h2>
            <small>vs previous actual</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- FOOTER ----------
st.markdown("---")

st.caption("Updated automatically | Data sources: National Records of Scotland")

