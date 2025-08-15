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
ACCENT_COLOR = "#ff7e5f"  # New accent color
BG_SIDEBAR = "#f8f9fa"
BG_MAIN = "#ffffff"
FONT = "Source Sans Pro"

# Custom CSS with enhanced styling
st.markdown(
    f"""
    <style>
    :root {{
        --primary: {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --accent: {ACCENT_COLOR};
    }}
    
    .main {{ 
        background-color: {BG_MAIN};
        background-image: radial-gradient(circle at 10% 20%, rgba(13,115,119,0.05) 0%, rgba(255,255,255,1) 90%);
    }}
    
    .css-1d391kg {{ 
        background-color: {BG_SIDEBAR};
        border-right: 1px solid rgba(0,0,0,0.1);
    }}
    
    .css-1v3fvcr {{ 
        font-family: {FONT};
    }}
    
    .metric-card {{
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: none;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, var(--primary), var(--secondary));
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}
    
    .metric-card h2, .metric-card h4 {{
        color: var(--primary) !important;
        margin-bottom: 0.5rem;
    }}
    
    .metric-card h2 {{
        font-size: 2.2rem;
        font-weight: 700;
    }}
    
    .metric-card small {{
        color: #666666 !important;
        font-size: 0.9rem;
    }}
    
    h1, h2, h3, h4 {{
        color: var(--primary);
        font-weight: 700;
    }}
    
    .stPlotlyChart {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    
    .stContainer {{
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }}
    
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--primary), transparent);
        margin: 2rem 0;
    }}
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
    actuals["Registration Year"] = pd.to_datetime(actuals["Registration Year"], format="%Y")
    return forecasts, actuals

df_forecasts, df_actuals = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🔍 Filters")
    st.markdown("---")
    
    # Add decorative element
    st.markdown(
        f'<div style="height: 4px; background: linear-gradient(to right, {PRIMARY_COLOR}, {SECONDARY_COLOR}); margin-bottom: 2rem;"></div>',
        unsafe_allow_html=True
    )
    
    nhs_boards = sorted(df_actuals["nhs_board"].unique())
    age_groups = sorted(df_actuals["age_group"].unique())
    
    selected_nhs_board = st.selectbox("NHS Board", nhs_boards)
    selected_age_group = st.selectbox("Maternal Age Group", age_groups)
    
    # Add a small info section
    st.markdown("---")
    st.markdown(
        f'<div style="font-size: 0.8rem; color: #666; padding: 0.5rem; background: rgba(13,115,119,0.05); border-radius: 8px;">'
        'Data source: National Records of Scotland'
        '</div>',
        unsafe_allow_html=True
    )

# ---------- FILTER DATA ----------
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

# Custom divider
st.markdown(
    f'<div style="height: 3px; background: linear-gradient(to right, {PRIMARY_COLOR}, {SECONDARY_COLOR}, {ACCENT_COLOR}); margin: 1.5rem 0;"></div>',
    unsafe_allow_html=True
)

# ---------- TIMESERIES ----------
with st.container():
    st.subheader("📈 Birth Rate Trend")
    
    fig = go.Figure()
    
    # Actual data with enhanced styling
    fig.add_trace(
        go.Scatter(
            x=df_actuals_filtered["Registration Year"],
            y=df_actuals_filtered["birth_rate"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=PRIMARY_COLOR, width=4),
            marker=dict(
                size=8,
                color=PRIMARY_COLOR,
                line=dict(width=1, color='white')
            ),
            hovertemplate="<b>%{x|%Y}</b><br>%{y:.2f} births<extra></extra>"
        )
    )
    
    # Forecast with enhanced styling
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=SECONDARY_COLOR, width=4, dash='dot'),
            marker=dict(
                size=8,
                symbol='diamond',
                color=SECONDARY_COLOR,
                line=dict(width=1, color='white')
            ),
            hovertemplate="<b>%{x|%Y}</b><br>%{y:.2f} births<extra></extra>"
        )
    )
    
    # Confidence interval with gradient fill
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat_upper"],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_forecasts_filtered["ds"],
            y=df_forecasts_filtered["yhat_lower"],
            fill='tonexty',
            fillcolor=f'rgba(20,160,133,0.2)',
            mode='lines',
            line=dict(width=0),
            name="Confidence Interval",
            hoverinfo='skip'
        )
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Birth Rate (per 1,000 women)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            mirror=True
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor='lightgray',
            mirror=True
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------- HEATMAP ----------
with st.container():
    st.subheader("🔥 Birth Rate Heatmap by Month")
    
    df_heatmap = (
        df_actuals_filtered.assign(year=lambda d: d["Registration Year"].dt.year)
        .assign(month=lambda d: d["Registration Year"].dt.month)
        .pivot_table(index="month", columns="year", values="birth_rate")
    )
    
    # Create custom color scale
    colorscale = [
        [0.0, "#e6f3f3"],
        [0.2, "#a8d8d8"],
        [0.4, "#6abdbd"],
        [0.6, "#3aa3a3"],
        [0.8, "#1d8989"],
        [1.0, PRIMARY_COLOR]
    ]
    
    fig_heatmap = px.imshow(
        df_heatmap,
        labels=dict(x="Year", y="Month", color="Birth Rate"),
        aspect="auto",
        color_continuous_scale=colorscale,
        text_auto=".1f"
    )
    
    fig_heatmap.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(
            side="top",
            tickangle=-45
        ),
        yaxis=dict(
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ),
        coloraxis_colorbar=dict(
            title="Birth Rate",
            thickness=15,
            len=0.8,
            yanchor="middle",
            y=0.5
        )
    )
    
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
            <h4>Latest Actual</h4>
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
            <h4>Latest Forecast</h4>
            <h2>{latest_forecast:.2f}</h2>
            <small>per 1,000 women</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    arrow = "↓" if yoy_change < 0 else "↑"
    color = "#ff4d4d" if yoy_change < 0 else "#4dff4d"
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>YoY Change</h4>
            <h2 style="color: {color};">{arrow} {abs(yoy_change):.2f}%</h2>
            <small>vs previous actual</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- FOOTER ----------
st.markdown(
    f'<div style="height: 3px; background: linear-gradient(to right, {PRIMARY_COLOR}, {SECONDARY_COLOR}, {ACCENT_COLOR}); margin: 2rem 0 1rem;"></div>',
    unsafe_allow_html=True
)

st.caption(
    f'<div style="text-align: center; font-size: 0.9rem; color: #666;">'
    'Updated automatically | Data sources: National Records of Scotland'
    '</div>',
    unsafe_allow_html=True
)

# Add a decorative element at the bottom
st.markdown(
    f'<div style="text-align: center; margin-top: 1rem;">'
    f'<span style="display: inline-block; width: 40px; height: 4px; background: {PRIMARY_COLOR}; margin: 0 5px;"></span>'
    f'<span style="display: inline-block; width: 40px; height: 4px; background: {SECONDARY_COLOR}; margin: 0 5px;"></span>'
    f'<span style="display: inline-block; width: 40px; height: 4px; background: {ACCENT_COLOR}; margin: 0 5px;"></span>'
    '</div>',
    unsafe_allow_html=True
)
