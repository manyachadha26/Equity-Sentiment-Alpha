"""
Stage 3: Backtesting Engine
Pulls price data with yfinance, merges with sentiment, and backtests
whether positive sentiment predicts positive next-day returns.

Key question: If yesterday's WSB sentiment was bullish, should you have bought?
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import time


# ─────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────

def fetch_prices(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    if not start:
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    if not end:
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"[Prices] Fetching {ticker.upper()} from {start} to {end}...")

    # Convert dates to timestamps for Yahoo Finance API
    import requests as req
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    for attempt in range(3):
        try:
            response = req.get(url, headers=headers, timeout=30)
            data = response.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            ohlcv = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "date": [datetime.utcfromtimestamp(t).date() for t in timestamps],
                "open":   ohlcv["open"],
                "high":   ohlcv["high"],
                "low":    ohlcv["low"],
                "close":  ohlcv["close"],
                "volume": ohlcv["volume"],
            })
            df = df.dropna(subset=["close"])
            break
        except Exception as e:
            print(f"[Prices] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    else:
        raise ValueError(f"Could not fetch price data for {ticker}")

    df["daily_return"] = df["close"].pct_change()
    df["next_day_return"] = df["daily_return"].shift(-1)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)
    df["next_day_up"] = (df["next_day_return"] > 0).astype(int)

    print(f"[Prices] {len(df)} trading days loaded.")
    return df


# ─────────────────────────────────────────────
# MERGE SENTIMENT + PRICES
# ─────────────────────────────────────────────

def build_signal_df(daily_sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merges daily sentiment with price data on date.
    Sentiment from day T is used to predict return on day T+1.
    This is the signal dataframe — the core of the backtest.
    """
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.date

    merged = pd.merge(prices, daily_sentiment, on="date", how="inner")

    # Lag sentiment by 1 day to avoid lookahead bias
    # (we already built this in: sentiment on day T predicts next_day_return on T+1)
    merged = merged.dropna(subset=["next_day_return", "mean_compound"])
    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"[Signal] Merged signal dataframe: {len(merged)} rows")
    return merged


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(signal_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    Simple long-only sentiment strategy:
      - Buy if compound sentiment > +threshold
      - Hold cash (0 return) otherwise

    Compares against buy-and-hold benchmark.

    threshold: minimum compound score to trigger a buy signal (tune this)
    """
    df = signal_df.copy()

    # Signal: 1 = buy, 0 = stay in cash
    df["signal"] = (df["weighted_compound"] > threshold).astype(int)

    # Strategy return: only take position when signal=1
    df["strategy_return"] = df["signal"].shift(1).fillna(0) * df["daily_return"]
    df["bh_return"] = df["daily_return"]  # buy and hold

    # Cumulative returns
    df["strategy_cumret"] = (1 + df["strategy_return"]).cumprod()
    df["bh_cumret"] = (1 + df["bh_return"]).cumprod()

    # ── Performance Metrics ──

    n_trades = int(df["signal"].sum())
    n_days = len(df)

    # Annualization factor (assume ~252 trading days/year)
    ann_factor = 252

    strategy_total = float(df["strategy_cumret"].iloc[-1]) - 1
    bh_total = float(df["bh_cumret"].iloc[-1]) - 1

    strategy_ann = float((1 + strategy_total) ** (ann_factor / n_days) - 1)
    bh_ann = float((1 + bh_total) ** (ann_factor / n_days) - 1)

    # Sharpe ratio (annualized, risk-free ≈ 0 for simplicity)
    strategy_sharpe = float(
        df["strategy_return"].mean() / (df["strategy_return"].std() + 1e-9) * np.sqrt(ann_factor)
    )
    bh_sharpe = float(
        df["bh_return"].mean() / (df["bh_return"].std() + 1e-9) * np.sqrt(ann_factor)
    )

    # Max drawdown
    def max_drawdown(cumret_series):
        rolling_max = cumret_series.cummax()
        drawdown = (cumret_series - rolling_max) / rolling_max
        return float(drawdown.min())

    strategy_mdd = max_drawdown(df["strategy_cumret"])
    bh_mdd = max_drawdown(df["bh_cumret"])

    # Win rate: on days we traded, how often were we right?
    traded = df[df["signal"].shift(1).fillna(0) == 1]
    win_rate = float((traded["daily_return"] > 0).mean()) if len(traded) > 0 else 0.0

    # Directional accuracy: does compound score direction match price direction?
    df["pred_direction"] = (df["weighted_compound"] > 0).astype(int)
    dir_accuracy = float((df["pred_direction"] == df["next_day_up"]).mean())

    # Correlation between compound score and next-day return
    corr = float(df["weighted_compound"].corr(df["next_day_return"]))

    metrics = {
        "ticker": df["ticker"].iloc[0] if "ticker" in df.columns else "N/A",
        "n_days": n_days,
        "n_trades": n_trades,
        "trade_pct": round(n_trades / n_days * 100, 1),
        "strategy_total_return": round(strategy_total * 100, 2),
        "bh_total_return": round(bh_total * 100, 2),
        "strategy_ann_return": round(strategy_ann * 100, 2),
        "bh_ann_return": round(bh_ann * 100, 2),
        "strategy_sharpe": round(strategy_sharpe, 3),
        "bh_sharpe": round(bh_sharpe, 3),
        "strategy_max_drawdown": round(strategy_mdd * 100, 2),
        "bh_max_drawdown": round(bh_mdd * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "directional_accuracy": round(dir_accuracy * 100, 2),
        "sentiment_price_correlation": round(corr, 4),
        "sentiment_threshold": threshold,
    }

    return {"metrics": metrics, "trades_df": df}


# ─────────────────────────────────────────────
# THRESHOLD SWEEP (find optimal threshold)
# ─────────────────────────────────────────────

def sweep_thresholds(signal_df: pd.DataFrame, thresholds: list = None) -> pd.DataFrame:
    """
    Runs the backtest across multiple sentiment thresholds.
    Useful for finding the optimal entry signal.
    Plots the sensitivity of strategy returns to threshold choice.
    """
    if thresholds is None:
        thresholds = np.arange(-0.2, 0.3, 0.05).round(2).tolist()

    rows = []
    for t in thresholds:
        result = run_backtest(signal_df, threshold=t)
        rows.append(result["metrics"])

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_backtest(df: pd.DataFrame, ticker: str, path: str = "data/backtest"):
    os.makedirs(path, exist_ok=True)
    fp = f"{path}/{ticker.upper()}_backtest.csv"
    df.to_csv(fp, index=False)
    print(f"[Saved] {fp}")
    return fp


# ─────────────────────────────────────────────
# QUICK TEST — run in Colab after Stage 2
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from sentiment_engine import load_sentiment

    TICKER = "TSLA"
    THRESHOLD = 0.05

    # Load daily sentiment from Stage 2
    daily_df = load_sentiment(TICKER, level="daily")

    # Fetch prices
    prices = fetch_prices(TICKER)

    # Build signal dataframe
    signal_df = build_signal_df(daily_df, prices)

    # Run backtest
    result = run_backtest(signal_df, threshold=THRESHOLD)
    metrics = result["metrics"]
    trades_df = result["trades_df"]

    print("\n─── Backtest Results ───")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    save_backtest(trades_df, TICKER)

    # Threshold sweep
    print("\n─── Threshold Sweep ───")
    sweep = sweep_thresholds(signal_df)
    print(sweep[["sentiment_threshold", "strategy_total_return", "bh_total_return",
                 "strategy_sharpe", "directional_accuracy"]].to_string(index=False))
