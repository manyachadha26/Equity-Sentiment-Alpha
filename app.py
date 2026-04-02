"""
Stage 4: Streamlit Dashboard
The full app — runs the pipeline and visualizes results.
Deploy to Streamlit Cloud by connecting your GitHub repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add parent dir so imports work from project root
sys.path.append(os.path.dirname(__file__))

from data_collector import collect_all
from sentiment_engine import score_dataframe, aggregate_daily_sentiment
from backtest_engine import fetch_prices, build_signal_df, run_backtest, sweep_thresholds


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Sentiment Alpha Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 2px; }
    .metric-value { font-size: 24px; font-weight: 600; }
    .green { color: #00C48C; }
    .red { color: #FF4B4B; }
    .neutral { color: #888; }
    .section-title { font-size: 18px; font-weight: 600; margin: 1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Sentiment Alpha")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="TSLA").upper().strip()
    reddit_limit = st.slider("Reddit Posts to Scrape", 50, 500, 150, step=50)
    sentiment_threshold = st.slider(
        "Buy Signal Threshold",
        min_value=-0.3, max_value=0.3, value=0.05, step=0.01,
        help="Minimum weighted compound score to trigger a long signal."
    )

    run_button = st.button("🚀 Run Pipeline", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Scrape Reddit (r/wsb) + Yahoo Finance news
    2. Score each post with FinBERT (finance-tuned BERT)
    3. Aggregate to daily weighted sentiment
    4. Backtest: buy when sentiment > threshold
    """)
    st.markdown("---")
    st.caption("Built with FinBERT · PRAW · yfinance · Streamlit")


# ─────────────────────────────────────────────
# PIPELINE RUNNER (cached by ticker + limit)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def run_pipeline(ticker: str, reddit_limit: int):
    raw_df = collect_all(ticker, reddit_limit=reddit_limit)
    if raw_df.empty:
        return None, None, None

    scored_df = score_dataframe(raw_df)
    daily_df = aggregate_daily_sentiment(scored_df)
    prices = fetch_prices(ticker)
    signal_df = build_signal_df(daily_df, prices)

    return scored_df, daily_df, signal_df


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

st.title(f"Equity Sentiment Alpha Engine")
st.markdown("*Does Reddit + news sentiment predict next-day stock returns?*")

if not run_button:
    st.info("👈 Configure your ticker in the sidebar and click **Run Pipeline** to start.")
    st.stop()

with st.spinner(f"Running pipeline for **{ticker}**... (FinBERT takes ~30s on CPU)"):
    scored_df, daily_df, signal_df = run_pipeline(ticker, reddit_limit)

if signal_df is None or signal_df.empty:
    st.error("No data returned. Try a more popular ticker like TSLA, NVDA, or AAPL.")
    st.stop()

result = run_backtest(signal_df, threshold=sentiment_threshold)
metrics = result["metrics"]
trades_df = result["trades_df"]


# ─────────────────────────────────────────────
# SECTION 1: KEY METRICS
# ─────────────────────────────────────────────

st.markdown(f"### {ticker} — Pipeline Results")

col1, col2, col3, col4, col5 = st.columns(5)

def metric_color(val, good_positive=True):
    if val > 0:
        return "green" if good_positive else "red"
    elif val < 0:
        return "red" if good_positive else "green"
    return "neutral"

with col1:
    v = metrics["strategy_total_return"]
    st.metric("Strategy Return", f"{v:+.1f}%", delta=f"{v - metrics['bh_total_return']:+.1f}% vs B&H")

with col2:
    st.metric("Buy & Hold Return", f"{metrics['bh_total_return']:+.1f}%")

with col3:
    st.metric("Sharpe Ratio", f"{metrics['strategy_sharpe']:.2f}",
              delta=f"B&H: {metrics['bh_sharpe']:.2f}")

with col4:
    st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%",
              help="% of days sentiment direction matched price direction")

with col5:
    st.metric("Sentiment–Return Corr.", f"{metrics['sentiment_price_correlation']:.3f}",
              help="Pearson correlation between daily sentiment score and next-day return")

st.markdown("---")


# ─────────────────────────────────────────────
# SECTION 2: CUMULATIVE RETURN CHART
# ─────────────────────────────────────────────

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("#### Cumulative Returns: Strategy vs Buy & Hold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df["date"], y=(trades_df["strategy_cumret"] - 1) * 100,
        name="Sentiment Strategy", line=dict(color="#00C48C", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=trades_df["date"], y=(trades_df["bh_cumret"] - 1) * 100,
        name="Buy & Hold", line=dict(color="#4A90D9", width=2, dash="dot"),
    ))
    # Shade signal regions
    buy_periods = trades_df[trades_df["signal"] == 1]
    for _, row in buy_periods.iterrows():
        fig.add_vrect(
            x0=str(row["date"]), x1=str(row["date"]),
            fillcolor="rgba(0,196,140,0.07)", line_width=0,
        )
    fig.update_layout(
        height=320, template="plotly_dark",
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Date",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("#### Trade Statistics")
    st.markdown(f"""
| Metric | Value |
|---|---|
| Days in backtest | {metrics['n_days']} |
| Days with signal | {metrics['n_trades']} ({metrics['trade_pct']}%) |
| Win rate | {metrics['win_rate']}% |
| Max drawdown (strat) | {metrics['strategy_max_drawdown']}% |
| Max drawdown (B&H) | {metrics['bh_max_drawdown']}% |
| Signal threshold | {metrics['sentiment_threshold']} |
""")


# ─────────────────────────────────────────────
# SECTION 3: DAILY SENTIMENT + PRICE
# ─────────────────────────────────────────────

st.markdown("#### Daily Sentiment Score vs Price")

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.65, 0.35], vertical_spacing=0.04)

# Price
fig2.add_trace(go.Candlestick(
    x=trades_df["date"],
    open=trades_df["open"], high=trades_df["high"],
    low=trades_df["low"], close=trades_df["close"],
    name="Price", increasing_line_color="#00C48C",
    decreasing_line_color="#FF4B4B",
), row=1, col=1)

# Sentiment bar
colors = ["#00C48C" if v > 0 else "#FF4B4B" for v in trades_df["weighted_compound"]]
fig2.add_trace(go.Bar(
    x=trades_df["date"], y=trades_df["weighted_compound"],
    name="Weighted Compound", marker_color=colors, opacity=0.8,
), row=2, col=1)

# Threshold line
fig2.add_hline(y=sentiment_threshold, row=2, col=1,
               line_dash="dot", line_color="yellow",
               annotation_text=f"threshold={sentiment_threshold}")

fig2.update_layout(
    height=480, template="plotly_dark",
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 4: THRESHOLD SENSITIVITY
# ─────────────────────────────────────────────

with st.expander("🔬 Threshold Sensitivity Analysis"):
    st.markdown("How sensitive is the strategy to the chosen sentiment threshold?")
    sweep = sweep_thresholds(signal_df)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sweep["sentiment_threshold"], y=sweep["strategy_total_return"],
        name="Strategy Return (%)", mode="lines+markers", line=dict(color="#00C48C"),
    ))
    fig3.add_hline(
        y=metrics["bh_total_return"], line_dash="dot", line_color="#4A90D9",
        annotation_text="Buy & Hold",
    )
    fig3.add_vline(x=sentiment_threshold, line_dash="dot", line_color="yellow",
                   annotation_text="Current")
    fig3.update_layout(
        height=300, template="plotly_dark",
        xaxis_title="Sentiment Threshold",
        yaxis_title="Total Return (%)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        sweep[["sentiment_threshold", "strategy_total_return", "bh_total_return",
               "strategy_sharpe", "directional_accuracy", "n_trades"]].round(2),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# SECTION 5: RAW POSTS EXPLORER
# ─────────────────────────────────────────────

with st.expander("📝 Raw Posts & Sentiment Scores"):
    show_df = scored_df[["source", "timestamp", "title", "label", "compound",
                          "positive", "negative", "neutral", "score"]].copy()
    show_df["timestamp"] = pd.to_datetime(show_df["timestamp"]).dt.strftime("%Y-%m-%d")

    label_filter = st.multiselect(
        "Filter by sentiment",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )
    filtered = show_df[show_df["label"].isin(label_filter)]
    st.dataframe(filtered.head(200), use_container_width=True)
