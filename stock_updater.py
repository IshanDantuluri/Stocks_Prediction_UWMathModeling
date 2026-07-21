import yfinance as yf
import pandas as pd
import requests
import os
import time
from datetime import datetime

EXCEL_FILE = "stocks.xlsx"
START_DATE = "2015-01-01"
EVENTS_START_DATE = "2026-07-14"  # GDELT DOC 2.0 API only reliably covers this onward
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# Increased delays to respect GDELT rate limits and server load
GDELT_DELAY = 15          # seconds between requests
GDELT_429_WAIT = 60       # seconds to wait after a 429 before retrying
MAX_RECORDS_PER_QUERY = 10

# Simplified keywords mapping to keep URL length short and lessen server query load
EVENT_KEYWORDS = {
    "plane_crash":       ["crash"],
    "earthquake":        ["earthquake"],
    "mass_shooting":     ["shooting"],
    "terrorist_attack":  ["attack", "terrorism"],
    "explosion":         ["explosion", "blast"],
    "wildfire":          ["wildfire"],
    "natural_disaster":  ["disaster", "flood", "hurricane"],
    "train_crash":       ["derailment"],
    "building_collapse": ["collapse"],
}

# Major outlets used as a proxy for "this was actually a big story"
TOP_NEWS_DOMAINS = ["reuters.com", "apnews.com", "bbc.co.uk", "cnn.com", "aljazeera.com"]

def build_keyword_query():
    # Simplified, concise OR search to prevent GDELT connection timeouts
    keywords = ["crash", "earthquake", "attack", "explosion", "disaster", "shooting", "wildfire", "derailment"]
    return "(" + " OR ".join(keywords) + ")"

def build_top_news_query():
    domains = " OR ".join(f"domain:{d}" for d in TOP_NEWS_DOMAINS)
    return f"({domains})"

KEYWORD_QUERY   = build_keyword_query()
TOP_NEWS_QUERY  = build_top_news_query()

def tag_category(title):
    title_lower = title.lower()
    for category, phrases in EVENT_KEYWORDS.items():
        for p in phrases:
            if p in title_lower:
                return category
    return "other"

# ── Stock helpers ──────────────────────────────────────────────────────────────

def load_existing():
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE, index_col=0, sheet_name="Stocks")
        df = df[pd.to_datetime(df.index, errors="coerce").notna()]
        df.index = pd.to_datetime(df.index)
        print(f"Loaded stocks: {list(df.columns)}")
        return df
    return pd.DataFrame()

def download_ticker(ticker):
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": ticker})
    df.index = pd.to_datetime(df.index)
    return df

# ── GDELT helpers ──────────────────────────────────────────────────────────────

def fetch_gdelt_artlist(query, day, sort=None, maxrecords=MAX_RECORDS_PER_QUERY, retries=3):
    """
    Query GDELT DOC 2.0 API in article-list mode for a single day.
    Returns a list of dicts: {title, url, domain, seendate}.
    Returns None on total failure (vs. [] which means it succeeded but found nothing).
    """
    day_start = day.strftime("%Y%m%d")
    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc?"
        f"query={requests.utils.quote(query)}"
        f"&mode=artlist"
        f"&maxrecords={maxrecords}"
        f"&startdatetime={day_start}000000&enddatetime={day_start}235959"
        f"&format=json"
    )
    if sort:
        url += f"&sort={sort}"

    for attempt in range(retries):
        try:
            time.sleep(GDELT_DELAY)
            # Timeout increased to 60s to prevent prematurely dropping long queries
            r = requests.get(url, timeout=60)

            if r.status_code == 429:
                print(f"      Rate limited (429), waiting {GDELT_429_WAIT}s...")
                time.sleep(GDELT_429_WAIT)
                continue

            if r.status_code != 200:
                print(f"      HTTP {r.status_code}: {r.text[:200]}")
                return None

            data = r.json()
            articles = data.get("articles", [])
            return [
                {
                    "title":    a.get("title", ""),
                    "url":      a.get("url", ""),
                    "domain":   a.get("domain", ""),
                    "seendate": a.get("seendate", ""),
                }
                for a in articles
            ]

        except Exception as e:
            print(f"      Attempt {attempt+1} failed: {e}")
            time.sleep(10)

    print(f"      Giving up after {retries} attempts.")
    return None

def build_events_features(start_date, end_date):
    """
    Pull day-by-day event headlines: keyword-matched incidents + major-outlet top news.
    Returns a flat DataFrame: date, category, title, domain, url.
    """
    print("\nFetching GDELT day-by-day events...")
    print(f"(Waiting {GDELT_DELAY}s between requests — this will take a while)\n")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rows = []

    for i, day in enumerate(dates, 1):
        day_str = day.strftime("%Y-%m-%d")
        print(f"  [{i}/{len(dates)}] {day_str}...")

        # Keyword-based specific incidents
        keyword_articles = fetch_gdelt_artlist(KEYWORD_QUERY, day)
        if keyword_articles:
            for a in keyword_articles:
                rows.append({
                    "date": day,
                    "category": tag_category(a["title"]),
                    "title": a["title"],
                    "domain": a["domain"],
                    "url": a["url"],
                })
            print(f"      keyword matches: {len(keyword_articles)}")

        # General top news from major outlets, sorted by most extreme tone
        top_articles = fetch_gdelt_artlist(TOP_NEWS_QUERY, day, sort="ToneDesc")
        if top_articles:
            existing_urls = {r["url"] for r in rows if r["date"] == day}
            for a in top_articles:
                if a["url"] in existing_urls:
                    continue  # skip duplicates already caught by keyword query
                rows.append({
                    "date": day,
                    "category": "top_news",
                    "title": a["title"],
                    "domain": a["domain"],
                    "url": a["url"],
                })
            print(f"      top news: {len(top_articles)}")

    return pd.DataFrame(rows)

def load_existing_events():
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE, sheet_name="Events")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].notna()]
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "category", "title", "domain", "url"])

# ── Save ───────────────────────────────────────────────────────────────────────

def save(stock_df, events_df):
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
        stock_df.to_excel(writer, sheet_name="Stocks")
        if not events_df.empty:
            events_df.to_excel(writer, sheet_name="Events", index=False)
    print(f"\nSaved to {EXCEL_FILE}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Stock + News Data Updater")
    print("=" * 50)

    stock_df = load_existing()
    events_df = load_existing_events()

    print("\nWhat do you want to do?")
    print("  1. Add new stock tickers")
    print("  2. Update news/events data (GDELT)")
    print("  3. Both")
    choice = input("\nEnter 1, 2, or 3: ").strip()

    # ── Add stocks ──
    if choice in ("1", "3"):
        print("\nEnter ticker symbols to add (one per line).")
        print("Press ENTER on an empty line when done.\n")
        new_tickers = []
        while True:
            t = input("Ticker: ").strip().upper()
            if t == "":
                break
            if t in stock_df.columns:
                print(f"  {t} already exists — skipping.")
            else:
                new_tickers.append(t)

        for ticker in new_tickers:
            try:
                new_df = download_ticker(ticker)
                if new_df.empty or new_df[ticker].isna().all():
                    print(f"  Skipped {ticker} — no data found.")
                    continue
                if stock_df.empty:
                    stock_df = new_df
                else:
                    stock_df = stock_df.join(new_df, how="outer")
                print(f"  Added {ticker}")
            except Exception as e:
                print(f"  Failed {ticker}: {e}")

    # ── Update events ──
    if choice in ("2", "3"):
        if events_df.empty:
            events_start = EVENTS_START_DATE
        else:
            last = events_df["date"].max()
            events_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if events_start < EVENTS_START_DATE:
            events_start = EVENTS_START_DATE

        if events_start <= END_DATE:
            days_to_fetch = len(pd.date_range(start=events_start, end=END_DATE, freq="D"))
            estimated_mins = round((days_to_fetch * 2 * GDELT_DELAY) / 60, 1)
            print(f"\nFetching {days_to_fetch} days x 2 queries each.")
            print(f"Estimated time: ~{estimated_mins} minutes (due to rate limiting).")
            print("Do not close this window.\n")

            new_events = build_events_features(events_start, END_DATE)
            if events_df.empty:
                events_df = new_events
            else:
                events_df = pd.concat([events_df, new_events], ignore_index=True)
            events_df = events_df.sort_values("date")
        else:
            print("Events data is already up to date.")

    # ── Save ──
    save(stock_df, events_df)

    print(f"\nStocks sheet:  {list(stock_df.columns)}")
    if not events_df.empty:
        print(f"Events sheet:  {len(events_df)} events, {events_df['date'].nunique()} days covered")
        print(f"Events range:  {events_df['date'].min().date()} → {events_df['date'].max().date()}")

if __name__ == "__main__":
    main()