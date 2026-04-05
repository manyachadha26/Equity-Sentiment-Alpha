# Equity-Sentiment-Alpha
# 📈 Equity Sentiment Alpha Engine

> I built a sentiment engine to find out if financial news actually predicts next-day stock returns. The finding was surprising.

**Live Demo:** https://equity-sentiment-alpha-5kjtunusir4pj3f3lycidr.streamlit.app/

---

## The Finding

Positive news sentiment was **inversely** predictive for TSLA — when media sentiment was bullish, the stock tended to drop the next day. Directional accuracy came in at 26.3%, worse than a coin flip, with a sentiment-return correlation of -0.28.

Buy the rumor, sell the news — confirmed by data.

---

## What it does
Yahoo Finance + Finviz + Google News
↓
FinBERT sentiment scoring
(positive / negative / neutral + compound score)
↓
Daily sentiment aggregation
↓
Backtest: buy when sentiment > threshold
↓
Streamlit dashboard: returns, Sharpe, drawdown, signal visualization

## Results (TSLA, 90-day backtest)

| Metric | Sentiment Strategy | Buy & Hold |
|---|---|---|
| Total Return | -1.1% | -0.5% |
| Sharpe Ratio | -0.81 | 0.04 |
| Max Drawdown | -4.44% | -12.19% |
| Directional Accuracy | 26.3% | 50% (random) |
| Sentiment-Return Correlation | -0.283 | — |

## Stack

| Component | Tool |
|---|---|
| News scraping | Yahoo Finance RSS, Finviz, Google News RSS |
| Sentiment model | ProsusAI/FinBERT (HuggingFace) |
| Price data | Yahoo Finance API |
| Backtesting | Custom engine (pandas/numpy) |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/manyachadha26/Equity-Sentiment-Alpha.git
cd Equity-Sentiment-Alpha
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app.py
```

### 3. Or open the live app
https://equity-sentiment-alpha-5kjtunusir4pj3f3lycidr.streamlit.app/

## Project structure
Equity-Sentiment-Alpha/
├── data_collector.py      # News scraping (Yahoo, Finviz, Google News)
├── sentiment_engine.py    # FinBERT scoring + daily aggregation
├── backtest_engine.py     # Price data + backtest logic
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── runtime.txt

## Key learnings
- FinBERT significantly outperforms VADER on financial text — it understands domain-specific language like "short squeeze", "beat estimates", "dilution"
- For TSLA specifically, positive sentiment is a contrarian signal — likely because heavily covered positive news is already priced in
- Sentiment signal works better as a filter than a directional predictor
- More data (6–12 months) would improve backtest reliability

---

*Project 1 of my #BuildInPublic challenge — building one real project every 3 days.*
*Follow along on X for daily updates.*
