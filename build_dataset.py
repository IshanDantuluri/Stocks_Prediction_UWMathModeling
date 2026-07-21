import os
from datetime import datetime
import pandas as pd
import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────────
EXCEL_FILE = "stocks.xlsx"
HISTORICAL_NEWS_CSV = "historical_news.csv"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# 5 Major Tickers per Sector
SECTOR_TICKERS = [
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Health Care
    "LLY", "JNJ", "ABBV", "UNH", "MRK",
    # Industrial
    "CAT", "GE", "RTX", "HON", "DE",
    # Materials
    "LIN", "APD", "FCX", "NEM", "SHW",
    # Tech
    "NVDA", "AAPL", "MSFT", "AVGO", "AMD"
]

# ── Stock Data Fetcher ────────────────────────────────────────────────────────
def fetch_stock_prices(tickers, start_date, end_date):
    print("\n[1/3] Downloading historical stock prices from Yahoo Finance...")
    stock_data = {}
    
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns:
                stock_data[ticker] = df["Close"]
            else:
                print(f"  Warning: No data found for {ticker}")
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")

    stocks_df = pd.DataFrame(stock_data)
    stocks_df.index = pd.to_datetime(stocks_df.index)
    stocks_df.sort_index(inplace=True)
    return stocks_df

# ── News Feature Processor ───────────────────────────────────────────────────
def process_news_features(csv_path):
    print("\n[2/3] Loading and engineering features from historical_news.csv...")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please run the BigQuery SQL script first.")
        return pd.DataFrame()

    news_df = pd.read_csv(csv_path)
    news_df["date"] = pd.to_datetime(news_df["date"])

    # Aggregate daily news features for Machine Learning training
    daily_news = news_df.groupby("date").agg(
        total_daily_events=("event_category", "count"),
        avg_daily_sentiment=("article_sentiment", "mean"),
        max_media_coverage=("media_coverage_volume", "max"),
        total_media_coverage=("media_coverage_volume", "sum"),
        avg_goldstein_impact=("goldstein_impact", "mean"),
        # Count frequency per event category
        disaster_count=("event_category", lambda x: (x == "disaster_accident").sum()),
        military_count=("event_category", lambda x: (x == "military_conflict").sum()),
        sanctions_count=("event_category", lambda x: (x == "coercion_sanctions").sum()),
        protest_count=("event_category", lambda x: (x == "protest_strike").sum()),
        attack_count=("event_category", lambda x: (x == "assault_attack").sum())
    ).reset_index()

    daily_news.set_index("date", inplace=True)
    daily_news.sort_index(inplace=True)
    return daily_news

# ── Save to Excel ─────────────────────────────────────────────────────────────
def save_data(stocks_df, news_df, raw_news_path):
    print("\n[3/3] Saving data to Excel...")
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        # Sheet 1: Stock Prices
        stocks_df.to_excel(writer, sheet_name="Stocks")
        
        # Sheet 2: Daily Engineered News Features (Joined by Date)
        if not news_df.empty:
            news_df.to_excel(writer, sheet_name="Daily_News_Features")
            
        # Sheet 3: Raw News Events
        if os.path.exists(raw_news_path):
            raw_df = pd.read_csv(raw_news_path)
            raw_df.head(10000).to_excel(writer, sheet_name="Raw_Events_Sample", index=False)

    print(f"\nSuccess! All data saved to '{EXCEL_FILE}'")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Stock Market + News Dataset Pipeline")
    print("=" * 60)

    # 1. Download Stock Data
    stocks_df = fetch_stock_prices(SECTOR_TICKERS, START_DATE, END_DATE)
    
    # 2. Process BigQuery News Data
    news_df = process_news_features(HISTORICAL_NEWS_CSV)

    # 3. Save Output
    save_data(stocks_df, news_df, HISTORICAL_NEWS_CSV)

    print("\nSummary:")
    print(f"  Stocks tracked: {len(stocks_df.columns)} tickers")
    print(f"  Stock data range: {stocks_df.index.min().strftime('%Y-%m-%d')} -> {stocks_df.index.max().strftime('%Y-%m-%d')}")
    if not news_df.empty:
        print(f"  News feature range: {news_df.index.min().strftime('%Y-%m-%d')} -> {news_df.index.max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()