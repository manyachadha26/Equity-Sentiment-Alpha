"""
Data Collector v2 — No API Keys Required
Sources:
  1. Yahoo Finance RSS feed
  2. Finviz news scraper
  3. Google News RSS feed

All three are free, require no authentication, and give high-quality
financial headlines — arguably better signal than WSB posts anyway.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_yahoo_finance(ticker: str) -> pd.DataFrame:
    """Pulls headlines from Yahoo Finance RSS feed. No API key needed."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    print(f"[Yahoo Finance] Fetching headlines for {ticker}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
    except Exception as e:
        print(f"[Yahoo Finance] Error: {e}")
        return pd.DataFrame()

    records = []
    for item in items:
        try:
            title = item.find("title").text.strip()
            desc = item.find("description")
            description = desc.text.strip() if desc else ""
            pub_date = item.find("pubDate")
            timestamp = pd.to_datetime(pub_date.text) if pub_date else datetime.utcnow()
            link = item.find("link")
            url_val = link.text.strip() if link else ""
            records.append({
                "source": "yahoo_finance",
                "ticker": ticker.upper(),
                "title": title,
                "text": description,
                "timestamp": timestamp,
                "url": url_val,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"[Yahoo Finance] {len(df)} headlines collected")
    return df


def scrape_finviz(ticker: str) -> pd.DataFrame:
    """Scrapes news table from Finviz quote page. No API key needed."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    print(f"[Finviz] Scraping news for {ticker}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        news_table = soup.find("table", id="news-table")
        if not news_table:
            print("[Finviz] News table not found")
            return pd.DataFrame()
        rows = news_table.find_all("tr")
    except Exception as e:
        print(f"[Finviz] Error: {e}")
        return pd.DataFrame()

    records = []
    current_date = None

    for row in rows:
        try:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_cell = cells[0].text.strip()
            title_cell = cells[1]

            if len(date_cell) > 8:
                try:
                    current_date = pd.to_datetime(date_cell, format="%b-%d-%y %I:%M%p")
                except Exception:
                    current_date = pd.to_datetime(date_cell, errors="coerce")
            else:
                if current_date is not None:
                    try:
                        time_part = pd.to_datetime(date_cell, format="%I:%M%p").time()
                        current_date = datetime.combine(current_date.date(), time_part)
                    except Exception:
                        pass

            link_tag = title_cell.find("a")
            if not link_tag:
                continue
            title = link_tag.text.strip()
            href = link_tag.get("href", "")
            source_tag = title_cell.find("span")
            source_name = source_tag.text.strip() if source_tag else "finviz"

            records.append({
                "source": f"finviz_{source_name.lower().replace(' ', '_')}",
                "ticker": ticker.upper(),
                "title": title,
                "text": "",
                "timestamp": current_date if current_date else datetime.utcnow(),
                "url": href,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"[Finviz] {len(df)} headlines collected")
    return df


def scrape_google_news(ticker: str, company_name: str = None) -> pd.DataFrame:
    """Pulls from Google News RSS. No API key needed."""
    query = f"{ticker} stock" if not company_name else f"{ticker} {company_name} stock"
    query_encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
    print(f"[Google News] Fetching news for '{query}'...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
    except Exception as e:
        print(f"[Google News] Error: {e}")
        return pd.DataFrame()

    records = []
    for item in items:
        try:
            title = item.find("title").text.strip()
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()
            pub_date = item.find("pubDate")
            timestamp = pd.to_datetime(pub_date.text) if pub_date else datetime.utcnow()
            source_tag = item.find("source")
            source_name = source_tag.text.strip() if source_tag else "google_news"
            records.append({
                "source": f"google_{source_name.lower().replace(' ', '_')}",
                "ticker": ticker.upper(),
                "title": title,
                "text": "",
                "timestamp": timestamp,
                "url": "",
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"[Google News] {len(df)} articles collected")
    return df


def collect_all(ticker: str, company_name: str = None, reddit_limit: int = None) -> pd.DataFrame:
    """
    Collects from Yahoo Finance, Finviz, and Google News.
    No API keys required. reddit_limit param ignored (kept for compatibility).
    """
    print(f"\n{'='*50}\nCollecting news for {ticker.upper()}\n{'='*50}")

    dfs = []

    yahoo_df = scrape_yahoo_finance(ticker)
    if not yahoo_df.empty:
        dfs.append(yahoo_df)
    time.sleep(1)

    finviz_df = scrape_finviz(ticker)
    if not finviz_df.empty:
        dfs.append(finviz_df)
    time.sleep(1)

    google_df = scrape_google_news(ticker, company_name)
    if not google_df.empty:
        dfs.append(google_df)

    if not dfs:
        print("\n[Collector] No data collected from any source.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["full_text"] = (combined["title"] + " " + combined["text"].fillna("")).str.strip()
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["date"] = combined["timestamp"].dt.date
    combined = combined.drop_duplicates(subset=["title"])
    combined = combined.dropna(subset=["title", "timestamp"])
    combined = combined.sort_values("timestamp", ascending=False).reset_index(drop=True)

    print(f"\nTotal unique headlines: {len(combined)}")
    print(combined["source"].value_counts().to_string())
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}\n")
    return combined


def save_raw(df: pd.DataFrame, ticker: str, path: str = "data/raw"):
    os.makedirs(path, exist_ok=True)
    fp = f"{path}/{ticker.upper()}_raw.csv"
    df.to_csv(fp, index=False)
    print(f"[Saved] {fp}")
    return fp


def load_raw(ticker: str, path: str = "data/raw") -> pd.DataFrame:
    fp = f"{path}/{ticker.upper()}_raw.csv"
    return pd.read_csv(fp, parse_dates=["timestamp"])


if __name__ == "__main__":
    df = collect_all("TSLA", company_name="Tesla")
    print(df[["source", "title", "timestamp"]].head(15).to_string(index=False))
    save_raw(df, "TSLA")
