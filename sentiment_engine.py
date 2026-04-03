"""
Stage 2: Sentiment Analysis with FinBERT
Runs ProsusAI/finbert on scraped text and produces daily aggregated sentiment scores.
FinBERT is fine-tuned on financial text — much better than VADER/TextBlob for this.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import os


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    """
    Downloads and caches FinBERT on first run (~440MB).
    Subsequent runs load from cache — fast.
    """
    print(f"[FinBERT] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[FinBERT] Running on: {device.upper()}")
    return tokenizer, model, device


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def predict_sentiment_batch(texts: list, tokenizer, model, device, batch_size: int = 32) -> list:
    """
    Runs FinBERT on a list of texts in batches.
    Returns a list of dicts: {label, positive, negative, neutral, compound}

    FinBERT labels: positive, negative, neutral
    compound = positive - negative (range: -1 to 1, like VADER)
    """
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="[FinBERT] Scoring batches"):
        batch = texts[i: i + batch_size]

        # Truncate to 512 tokens (FinBERT max)
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        for prob in probs:
            pos, neg, neu = float(prob[0]), float(prob[1]), float(prob[2])
            results.append({
                "label": ["positive", "negative", "neutral"][np.argmax(prob)],
                "positive": round(pos, 4),
                "negative": round(neg, 4),
                "neutral": round(neu, 4),
                "compound": round(pos - neg, 4),  # synthetic compound score
            })

    return results


# ─────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw collected DataFrame, runs FinBERT on 'full_text',
    and returns the DataFrame with sentiment columns added.
    """
    tokenizer, model, device = load_finbert()

    texts = df["full_text"].fillna("").tolist()
    sentiment_results = predict_sentiment_batch(texts, tokenizer, model, device)

    sentiment_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

    print(f"[Scorer] Sentiment distribution:\n{df['label'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# DAILY AGGREGATION
# This is the key signal: aggregate sentiment per day per ticker
# We weight by Reddit score (upvotes) for Reddit posts, equal weight for news
# ─────────────────────────────────────────────

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates sentiment scores to daily level.
    Produces:
      - mean_compound: average sentiment across all posts that day
      - weighted_compound: upvote-weighted sentiment (Reddit posts)
      - sentiment_volume: how many posts/articles that day
      - bull_bear_ratio: positive / (positive + negative)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    # Fill missing scores (news articles) with 1 for equal weighting
    df["weight"] = 1

    def weighted_mean(group, col, weight_col):
        weights = group[weight_col]
        return (group[col] * weights).sum() / weights.sum()

    daily = []
    for (ticker, date), group in df.groupby(["ticker", "date"]):
        pos_count = (group["label"] == "positive").sum()
        neg_count = (group["label"] == "negative").sum()
        total = len(group)

        daily.append({
            "ticker": ticker,
            "date": date,
            "mean_compound": round(group["compound"].mean(), 4),
            "weighted_compound": round(weighted_mean(group, "compound", "weight"), 4),
            "mean_positive": round(group["positive"].mean(), 4),
            "mean_negative": round(group["negative"].mean(), 4),
            "mean_neutral": round(group["neutral"].mean(), 4),
            "sentiment_volume": total,
            "bull_bear_ratio": round(pos_count / (pos_count + neg_count + 1e-9), 4),
            "dominant_label": group["label"].mode()[0],
        })

    daily_df = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
    print(f"[Aggregator] Daily sentiment shape: {daily_df.shape}")
    return daily_df


# ─────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────

def save_sentiment(df: pd.DataFrame, ticker: str, level: str = "post", path: str = "data/sentiment"):
    os.makedirs(path, exist_ok=True)
    fp = f"{path}/{ticker.upper()}_{level}_sentiment.csv"
    df.to_csv(fp, index=False)
    print(f"[Saved] {fp}")
    return fp


def load_sentiment(ticker: str, level: str = "post", path: str = "data/sentiment") -> pd.DataFrame:
    fp = f"{path}/{ticker.upper()}_{level}_sentiment.csv"
    return pd.read_csv(fp, parse_dates=["timestamp"] if level == "post" else ["date"])


# ─────────────────────────────────────────────
# QUICK TEST — run in Colab after Stage 1
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_collector import load_raw

    TICKER = "TSLA"

    # Load raw data from Stage 1
    raw_df = load_raw(TICKER)

    # Score with FinBERT
    scored_df = score_dataframe(raw_df)
    save_sentiment(scored_df, TICKER, level="post")

    # Aggregate to daily
    daily_df = aggregate_daily_sentiment(scored_df)
    save_sentiment(daily_df, TICKER, level="daily")

    print("\nSample daily sentiment:")
    print(daily_df.tail(10).to_string(index=False))
