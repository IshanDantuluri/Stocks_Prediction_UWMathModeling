#!/usr/bin/env python
# coding: utf-8

# In[5]:


import copy
import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import scipy.stats as stats

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ---------------------------------------------------------
# 1. FEATURE LISTS
# ---------------------------------------------------------
LLM_FEATURE_NAMES = [
    'news_signed_impact', 'news_max_absolute_impact', 'news_confidence',
    'news_novelty', 'news_persistence', 'news_uncertainty_change',
    'news_disagreement', 'news_article_count', 'news_unique_event_count',
    'news_source_count', 'earnings_impact', 'regulatory_impact',
    'litigation_impact', 'product_impact', 'management_impact',
    'macroeconomic_impact', 'geopolitical_impact', 'revenue_channel_impact',
    'cost_channel_impact', 'supply_chain_channel_impact', 'demand_channel_impact',
    'reported_fact_count', 'analysis_count', 'speculation_count'
]

NEWS_SCOPES = ('ticker', 'sector', 'market')
SCOPED_LLM_FEATURE_NAMES = [
    f'{scope}__{feature}'
    for scope in NEWS_SCOPES
    for feature in LLM_FEATURE_NAMES
]
COUNT_FEATURE_NAMES = {
    'news_article_count', 'news_unique_event_count', 'news_source_count',
    'reported_fact_count', 'analysis_count', 'speculation_count',
}
UNIT_INTERVAL_FEATURE_NAMES = {
    'news_max_absolute_impact', 'news_confidence', 'news_novelty',
    'news_persistence', 'news_disagreement',
}
NEWS_COUNT_CAP = 5.0

QUANT_FEATURE_NAMES = [
    'log_return_1d', 'log_return_5d', 'log_return_21d',
    'intraday_range', 'overnight_gap', 'volume_zscore_20d',
    'volatility_20d', 'sma_ratio_20_200',
    'rsi_14', 'macd_hist', 'roc_3d', 'roc_10d', 'vol_regime_ratio',
    'log_dollar_volume_z', 'day_of_week', 'is_month_start', 'is_month_end',
    'sector_relative_return', 'volume_zscore_rank', 'return_5d_rank',
]

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
TICKERS_CSV = os.path.join(SCRIPT_DIR, "sp500_tickers.csv")

DATA_START = "2015-01-01"
DATA_END = "2026-01-01"
TRAIN_END = "2023-01-01"
VAL_END = "2025-01-01"

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def engineer_quant_features(df):
    df = df.copy()

    df['log_return_1d'] = df.groupby('Ticker')['Close'].pct_change(fill_method=None).apply(lambda x: np.log1p(x))
    df['log_return_5d'] = df.groupby('Ticker')['Close'].pct_change(5, fill_method=None).apply(lambda x: np.log1p(x))
    df['log_return_21d'] = df.groupby('Ticker')['Close'].pct_change(21, fill_method=None).apply(lambda x: np.log1p(x))

    df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
    df['overnight_gap'] = (df['Open'] - df.groupby('Ticker')['Close'].shift(1)) / (df.groupby('Ticker')['Close'].shift(1) + 1e-8)

    vol_mean = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    vol_std = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['volume_zscore_20d'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
    df['volatility_20d'] = df.groupby('Ticker')['log_return_1d'].transform(lambda x: x.rolling(20, min_periods=5).std())

    sma_20 = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    sma_200 = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200, min_periods=20).mean())
    df['sma_ratio_20_200'] = (sma_20 / (sma_200 + 1e-8)) - 1.0

    def rsi_14(close):
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=5).mean()
        avg_loss = loss.rolling(14, min_periods=5).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))
    df['rsi_14'] = df.groupby('Ticker')['Close'].transform(rsi_14)
    df['rsi_14'] = (df['rsi_14'] - 50.0) / 50.0

    def macd_hist(close):
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return (macd_line - signal_line) / (close + 1e-8)
    df['macd_hist'] = df.groupby('Ticker')['Close'].transform(macd_hist)

    df['roc_3d'] = df.groupby('Ticker')['Close'].pct_change(3, fill_method=None)
    df['roc_10d'] = df.groupby('Ticker')['Close'].pct_change(10, fill_method=None)

    vol_5d = df.groupby('Ticker')['log_return_1d'].transform(lambda x: x.rolling(5, min_periods=3).std())
    vol_60d = df.groupby('Ticker')['log_return_1d'].transform(lambda x: x.rolling(60, min_periods=20).std())
    df['vol_regime_ratio'] = (vol_5d / (vol_60d + 1e-8)) - 1.0

    df['dollar_volume'] = df['Close'] * df['Volume']
    df['log_dollar_volume'] = np.log1p(df['dollar_volume'])
    dv_mean = df.groupby('Ticker')['log_dollar_volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    dv_std = df.groupby('Ticker')['log_dollar_volume'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['log_dollar_volume_z'] = (df['log_dollar_volume'] - dv_mean) / (dv_std + 1e-8)

    df['day_of_week'] = df['Date'].dt.dayofweek / 4.0 - 0.5
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(float)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(float)

    # The prediction keyed by trade date T is made before T's open. Its label must
    # therefore contain only the return attainable after that open.
    df['raw_target_1d'] = df['Close'] / df['Open'] - 1.0

    return df


def engineer_cross_sectional_features(df):
    """Requires a Sector column already merged in. Run after all tickers combined."""
    df = df.copy()

    sector_avg_return = df.groupby(['Date', 'Sector'])['log_return_1d'].transform('mean')
    df['sector_relative_return'] = df['log_return_1d'] - sector_avg_return

    df['volume_zscore_rank'] = df.groupby('Date')['volume_zscore_20d'].rank(pct=True) - 0.5
    df['return_5d_rank'] = df.groupby('Date')['log_return_5d'].rank(pct=True) - 0.5

    return df


def compute_market_benchmark(start, end, retries=3):
    print("  Downloading SPY as market benchmark...")
    spy = None
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            candidate = yf.download(
                'SPY',
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
                timeout=30,
            )
            if not candidate.empty:
                spy = candidate
                break
            last_error = RuntimeError("Yahoo returned an empty SPY frame")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(attempt)
    if spy is None:
        raise RuntimeError(
            f"failed to download SPY after {retries} attempts: {last_error}"
        )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[['Open', 'Close']].reset_index()
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy['market_return_1d'] = spy['Close'] / spy['Open'] - 1.0
    return spy[['Date', 'market_return_1d']]


def _read_news_features(path):
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def merge_scoped_news_data(
    stock_quant_df,
    tickers_df,
    news_path=None,
    model_id=None,
    prompt_version=None,
):
    """Join leakage-safe trade-date rows and keep ticker/sector/market distinct.

    ``news_reasoning.py export-trading`` assigns event-date D information to the
    first market session T strictly after D. For a sample keyed by T, quantitative
    history must end at T-1, while these news values are available at T's open.
    """
    merged = stock_quant_df.copy()
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()

    if not news_path:
        print("No trading-date news file supplied. Adding zero placeholders.")
        for column in SCOPED_LLM_FEATURE_NAMES:
            merged[column] = 0.0
        merged[QUANT_FEATURE_NAMES] = merged[QUANT_FEATURE_NAMES].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        return merged

    print(f"Loading leakage-safe news features from {news_path}...")
    news = _read_news_features(news_path)
    required = {'trade_date', 'scope', 'entity_id', *LLM_FEATURE_NAMES}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"news feature file is missing columns: {sorted(missing)}")

    if model_id is not None and 'model_id' not in news:
        raise ValueError("news feature file has no model_id column")
    if prompt_version is not None and 'prompt_version' not in news:
        raise ValueError("news feature file has no prompt_version column")
    if model_id is not None:
        news = news[news['model_id'] == model_id]
    if prompt_version is not None:
        news = news[news['prompt_version'] == prompt_version]
    if news.empty:
        raise ValueError("no news rows remain after model/prompt filtering")

    versions = [
        column for column in ('model_id', 'prompt_version')
        if column in news and news[column].nunique(dropna=False) > 1
    ]
    if versions:
        raise ValueError(
            "news file contains multiple versions; select one with "
            f"--news-model-id/--news-prompt-version ({', '.join(versions)})"
        )

    news = news.copy()
    news['trade_date'] = pd.to_datetime(news['trade_date']).dt.normalize()
    duplicate_keys = news.duplicated(['trade_date', 'scope', 'entity_id'], keep=False)
    if duplicate_keys.any():
        raise ValueError(
            "news file has duplicate (trade_date, scope, entity_id) rows after filtering"
        )

    # Preserve the semantic zero point. Counts use the agreed saturating mapping:
    # 0 -> 0 and 5 or more -> 1. Signed/bounded judgments are not standardized.
    for feature in LLM_FEATURE_NAMES:
        values = pd.to_numeric(news[feature], errors='coerce').fillna(0.0)
        if feature in COUNT_FEATURE_NAMES:
            values = values.clip(0.0, NEWS_COUNT_CAP) / NEWS_COUNT_CAP
        elif feature in UNIT_INTERVAL_FEATURE_NAMES:
            values = values.clip(0.0, 1.0)
        else:
            values = values.clip(-1.0, 1.0)
        news[feature] = values

    ticker_sector = tickers_df[['Symbol', 'GICS Sector']].drop_duplicates('Symbol')
    merged = merged.merge(
        ticker_sector,
        left_on='Ticker',
        right_on='Symbol',
        how='left',
        validate='many_to_one',
    ).drop(columns='Symbol')

    def join_scope(frame, scope, left_entity):
        scoped = news[news['scope'] == scope][
            ['trade_date', 'entity_id', *LLM_FEATURE_NAMES]
        ].rename(columns={
            'trade_date': 'Date',
            **{feature: f'{scope}__{feature}' for feature in LLM_FEATURE_NAMES},
        })
        return frame.merge(
            scoped,
            left_on=['Date', left_entity],
            right_on=['Date', 'entity_id'],
            how='left',
            validate='many_to_one',
        ).drop(columns='entity_id')

    merged = join_scope(merged, 'ticker', 'Ticker')
    merged = join_scope(merged, 'sector', 'GICS Sector')

    market = news[news['scope'] == 'market']
    market_entities_per_day = market.groupby('trade_date')['entity_id'].nunique()
    if (market_entities_per_day > 1).any():
        raise ValueError("market scope contains more than one entity on a trade date")
    market = market[['trade_date', *LLM_FEATURE_NAMES]].rename(columns={
        'trade_date': 'Date',
        **{feature: f'market__{feature}' for feature in LLM_FEATURE_NAMES},
    })
    merged = merged.merge(market, on='Date', how='left', validate='many_to_one')

    merged[SCOPED_LLM_FEATURE_NAMES] = merged[
        SCOPED_LLM_FEATURE_NAMES
    ].fillna(0.0)
    merged[QUANT_FEATURE_NAMES] = merged[QUANT_FEATURE_NAMES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return merged

# ---------------------------------------------------------
# 3. TWO-BRANCH ARCHITECTURE
# ---------------------------------------------------------
class DirectionalBCELoss(nn.Module):
    """
    Binary cross-entropy on direction (beat market / not). There's no
    "predict near zero" shortcut here — the model has to commit to a real
    probability, or it gets penalized.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logit, continuous_target):
        binary_target = (continuous_target > 0).float()
        return self.bce(logit, binary_target)


class QuantEncoder(nn.Module):
    """Reads the 30-day technical sequence: GRU + attention."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        return context  # (batch, hidden_dim)


class NewsEncoder(nn.Module):
    """Reads only the most recent day's news/event features — a small feedforward net."""
    def __init__(self, input_dim, hidden_dim=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (batch, hidden_dim)


class TwoBranchModel(nn.Module):
    def __init__(
        self,
        quant_dim,
        news_dim,
        quant_hidden=64,
        news_hidden=16,
        dropout=0.2,
        use_news=True,
    ):
        super().__init__()
        self.use_news = use_news
        self.quant_encoder = QuantEncoder(quant_dim, hidden_dim=quant_hidden, dropout=dropout)
        self.news_encoder = (
            NewsEncoder(news_dim, hidden_dim=news_hidden, dropout=dropout)
            if use_news else None
        )
        fusion_dim = quant_hidden + (news_hidden if use_news else 0)
        self.fc1 = nn.Linear(fusion_dim, 32)
        self.layer_norm = nn.LayerNorm(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, quant_seq, news_today):
        quant_ctx = self.quant_encoder(quant_seq)
        if self.use_news:
            news_ctx = self.news_encoder(news_today)
            fused = torch.cat([quant_ctx, news_ctx], dim=1)
        else:
            fused = quant_ctx
        out = self.fc1(fused)
        out = self.layer_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out  # raw logit


class StockSequenceStore:
    """Store each ticker once so overlapping 30-day windows are sliced lazily."""
    def __init__(self, df, quant_cols, news_cols, target_col='target_alpha_1d'):
        self.groups = []
        for ticker, group in df.sort_values(['Ticker', 'Date']).groupby(
            'Ticker', sort=False
        ):
            self.groups.append({
                'ticker': ticker,
                'dates': group['Date'].to_numpy(dtype='datetime64[D]'),
                'quant': group[quant_cols].to_numpy(dtype=np.float32),
                'news': group[news_cols].to_numpy(dtype=np.float32),
                'target': group[target_col].to_numpy(dtype=np.float32),
            })


class MultiStockDataset(Dataset):
    """
    For trade date T, produces prices through T-1, news available at T's open,
    and T's open-to-close alpha. Split membership is based on T, while the
    quantitative lookback may legitimately cross a split boundary.
    """
    def __init__(self, store, start=None, end=None, seq_len=30):
        self.store = store
        self.seq_len = seq_len
        start_day = np.datetime64(start, 'D') if start else None
        end_day = np.datetime64(end, 'D') if end else None
        self.samples = []

        for group_index, group in enumerate(store.groups):
            for row_index in range(seq_len, len(group['dates'])):
                trade_day = group['dates'][row_index]
                if start_day is not None and trade_day < start_day:
                    continue
                if end_day is not None and trade_day >= end_day:
                    continue
                if not np.isfinite(group['target'][row_index]):
                    continue
                self.samples.append((group_index, row_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group_index, row_index = self.samples[idx]
        group = self.store.groups[group_index]
        quant_seq = group['quant'][row_index - self.seq_len:row_index]
        news_today = group['news'][row_index]
        target = np.array([group['target'][row_index]], dtype=np.float32)
        trade_day = group['dates'][row_index].astype(np.int64)
        return quant_seq, news_today, target, trade_day

def evaluate(model, loader):
    model.eval()
    logits, actuals, trade_days = [], [], []
    with torch.no_grad():
        for quant_seq, news_today, batch_y, batch_days in loader:
            quant_seq, news_today = quant_seq.to(device), news_today.to(device)
            p = model(quant_seq, news_today)
            logits.extend(p.detach().cpu().numpy().reshape(-1))
            actuals.extend(batch_y.numpy().reshape(-1))
            trade_days.extend(batch_days.numpy().reshape(-1))
    logits, actuals = np.array(logits), np.array(actuals)
    trade_days = np.array(trade_days)
    probs = 1 / (1 + np.exp(-logits))  # convert logits to probabilities
    pred_direction = np.where(probs > 0.5, 1, -1)
    actual_direction = np.sign(actuals)
    hit_rate = (pred_direction == actual_direction).mean()
    brier = np.mean((probs - (actuals > 0).astype(float)) ** 2)
    daily_ics, daily_spreads = [], []
    for trade_day in np.unique(trade_days):
        mask = trade_days == trade_day
        if mask.sum() < 3 or np.ptp(probs[mask]) == 0:
            continue
        daily_ic = stats.spearmanr(probs[mask], actuals[mask]).statistic
        if np.isfinite(daily_ic):
            daily_ics.append(daily_ic)
        if mask.sum() >= 20:
            order = np.argsort(probs[mask])
            tail = max(1, mask.sum() // 10)
            day_targets = actuals[mask]
            daily_spreads.append(
                day_targets[order[-tail:]].mean()
                - day_targets[order[:tail]].mean()
            )

    daily_ics = np.asarray(daily_ics)
    mean_daily_ic = daily_ics.mean() if len(daily_ics) else np.nan
    ic_std = daily_ics.std(ddof=1) if len(daily_ics) > 1 else np.nan
    icir = mean_daily_ic / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
    ic_p_value = (
        stats.ttest_1samp(daily_ics, 0.0).pvalue
        if len(daily_ics) > 1 else np.nan
    )
    return {
        'hit_rate': hit_rate,
        'brier': brier,
        'prob_std': probs.std(),
        'positive_rate': (actuals > 0).mean(),
        'mean_daily_ic': mean_daily_ic,
        'daily_ic_std': ic_std,
        'icir': icir,
        'ic_p_value': ic_p_value,
        'daily_ic_days': len(daily_ics),
        'mean_daily_decile_spread': (
            np.mean(daily_spreads) if daily_spreads else np.nan
        ),
    }


def _print_metrics(label, metrics):
    print(f"\n--- {label} ---")
    print(f"Directional Hit Rate: {metrics['hit_rate'] * 100:.2f}%")
    print(f"Positive-class Base Rate: {metrics['positive_rate'] * 100:.2f}%")
    print(
        f"Mean Daily Cross-sectional IC: {metrics['mean_daily_ic']:.4f} "
        f"across {metrics['daily_ic_days']:,} sessions"
    )
    print(f"Daily ICIR: {metrics['icir']:.4f}")
    print(f"Daily-IC t-test p-value: {metrics['ic_p_value']:.4g}")
    print(f"Brier Score: {metrics['brier']:.6f}")
    print(f"Prediction Prob Std Dev: {metrics['prob_std']:.6f}")
    print(
        "Mean Daily Top-minus-bottom Decile Alpha "
        f"(before costs): {metrics['mean_daily_decile_spread']:.6f}"
    )


def _reshape_yahoo_download(download_data, yahoo_to_canonical):
    if download_data is None or download_data.empty:
        return pd.DataFrame(
            columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
    if not isinstance(download_data.columns, pd.MultiIndex):
        if len(yahoo_to_canonical) != 1:
            raise ValueError("Yahoo returned non-MultiIndex data for multiple tickers")
        yahoo_symbol = next(iter(yahoo_to_canonical))
        raw = download_data.reset_index()
        raw['YahooTicker'] = yahoo_symbol
    else:
        raw = (
            download_data.stack(level=0, future_stack=True)
            .rename_axis(index=['Date', 'YahooTicker'])
            .reset_index()
        )
    raw['Ticker'] = raw['YahooTicker'].map(yahoo_to_canonical)
    raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
    columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    raw = raw[columns].dropna(subset=['Ticker', 'Open', 'High', 'Low', 'Close'])
    raw['Volume'] = raw['Volume'].fillna(0.0)
    return raw.sort_values(['Ticker', 'Date']).drop_duplicates(
        ['Ticker', 'Date'], keep='last'
    ).reset_index(drop=True)


def download_price_history(
    tickers_df,
    start,
    end,
    batch_size=25,
    retries=3,
):
    canonical_symbols = tickers_df['Symbol'].drop_duplicates().tolist()
    yahoo_to_canonical = {
        symbol.replace('.', '-'): symbol for symbol in canonical_symbols
    }
    yahoo_symbols = list(yahoo_to_canonical)
    frames = []
    total_batches = (len(yahoo_symbols) + batch_size - 1) // batch_size
    for batch_number, offset in enumerate(
        range(0, len(yahoo_symbols), batch_size), start=1
    ):
        pending = yahoo_symbols[offset:offset + batch_size]
        batch_frames = []
        for attempt in range(1, retries + 1):
            if not pending:
                break
            print(
                f"  Price batch {batch_number}/{total_batches}, attempt "
                f"{attempt}/{retries}: {len(pending)} ticker(s)"
            )
            try:
                downloaded = yf.download(
                    tickers=pending,
                    start=start,
                    end=end,
                    group_by='ticker',
                    progress=False,
                    threads=min(8, len(pending)),
                    auto_adjust=True,
                    timeout=30,
                )
                parsed = _reshape_yahoo_download(
                    downloaded,
                    {symbol: yahoo_to_canonical[symbol] for symbol in pending},
                )
            except Exception as exc:
                print(f"    Batch attempt failed: {type(exc).__name__}: {exc}")
                parsed = pd.DataFrame()
            if not parsed.empty:
                batch_frames.append(parsed)
                completed = {
                    canonical.replace('.', '-')
                    for canonical in parsed['Ticker'].unique()
                }
                pending = [
                    symbol for symbol in pending if symbol not in completed
                ]
            if pending and attempt < retries:
                time.sleep(attempt)
        frames.extend(batch_frames)

    raw = (
        pd.concat(frames, ignore_index=True)
        if frames else
        pd.DataFrame(
            columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
    )
    raw = raw.sort_values(['Ticker', 'Date']).drop_duplicates(
        ['Ticker', 'Date'], keep='last'
    ).reset_index(drop=True)
    missing = set(canonical_symbols) - set(raw['Ticker'].unique())
    if missing:
        print(
            f"  Warning: no data returned for {len(missing)} tickers: "
            f"{sorted(missing)}"
        )
    return raw


def load_or_download_price_history(
    tickers_df,
    start,
    end,
    cache_path=None,
    batch_size=25,
    retries=3,
    minimum_coverage=0.98,
):
    expected = set(tickers_df['Symbol'].drop_duplicates())
    columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    cached = pd.DataFrame(columns=columns)
    if cache_path and Path(cache_path).exists():
        print(f"\n[2/6] Loading cached prices from {cache_path}...")
        cached = pd.read_csv(cache_path, parse_dates=['Date'])
        missing_columns = set(columns) - set(cached.columns)
        if missing_columns:
            raise ValueError(
                f"price cache is missing columns: {sorted(missing_columns)}"
            )
        cached = cached[columns].dropna(
            subset=['Ticker', 'Open', 'High', 'Low', 'Close']
        )
        cached = cached[cached['Ticker'].isin(expected)]
        cached = cached.drop_duplicates(['Ticker', 'Date'], keep='last')

    covered = set(cached['Ticker'].unique())
    missing = expected - covered
    if missing:
        print(
            f"\n[2/6] Fetching {len(missing):,} missing ticker histories "
            f"in batches of {batch_size}..."
        )
        missing_df = tickers_df[tickers_df['Symbol'].isin(missing)]
        fetched = download_price_history(
            missing_df,
            start,
            end,
            batch_size=batch_size,
            retries=retries,
        )
        cached = pd.concat([cached, fetched], ignore_index=True)
        cached = cached.drop_duplicates(['Ticker', 'Date'], keep='last')

    covered = set(cached['Ticker'].unique())
    coverage = len(covered) / len(expected) if expected else 0.0
    missing = expected - covered
    print(
        f"  Valid price coverage: {len(covered):,}/{len(expected):,} "
        f"tickers ({coverage:.1%})"
    )
    if cache_path:
        cached.sort_values(['Ticker', 'Date']).to_csv(cache_path, index=False)
        print(f"  Wrote validated price cache to {cache_path}")
    if coverage < minimum_coverage:
        preview = ', '.join(sorted(missing)[:25])
        raise RuntimeError(
            f"price coverage {coverage:.1%} is below required "
            f"{minimum_coverage:.1%}; still missing {len(missing)} ticker(s): "
            f"{preview}"
        )
    if missing:
        print(
            f"  Continuing with {len(missing)} unresolved ticker(s) below "
            "the allowed coverage threshold."
        )
    return cached.sort_values(['Ticker', 'Date']).reset_index(drop=True)


def train_model(args):
    print(f"Using device: {device}")
    print("[1/6] Loading S&P 500 ticker list...")
    tickers_df = pd.read_csv(args.tickers)
    required_ticker_columns = {'Symbol', 'GICS Sector'}
    if missing := required_ticker_columns - set(tickers_df.columns):
        raise ValueError(f"ticker file is missing columns: {sorted(missing)}")
    sector_map = dict(zip(tickers_df['Symbol'], tickers_df['GICS Sector']))
    print(
        f"  Loaded {tickers_df['Symbol'].nunique()} tickers across "
        f"{tickers_df['GICS Sector'].nunique()} sectors."
    )
    print(
        "  Warning: this is the current S&P 500 membership, so historical "
        "results retain survivorship bias."
    )
    if args.model_mode == 'quant-news' and not args.news_features:
        raise ValueError(
            "--news-features is required for --model-mode quant-news; "
            "use --model-mode quant-only for the no-news ablation"
        )

    yfinance_cache = Path(args.yfinance_cache)
    yfinance_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yfinance_cache))
    raw_df = load_or_download_price_history(
        tickers_df,
        args.data_start,
        args.data_end,
        cache_path=args.price_cache,
        batch_size=args.download_batch_size,
        retries=args.download_retries,
        minimum_coverage=args.minimum_price_coverage,
    )

    raw_df['Sector'] = raw_df['Ticker'].map(sector_map)
    raw_df = raw_df.dropna(subset=['Sector'])

    print("\n[3/6] Engineering quantitative metrics...")
    quant_df = engineer_quant_features(raw_df)
    quant_df = engineer_cross_sectional_features(quant_df)

    print("\n[4/6] Computing same-session open-to-close alpha target...")
    spy_df = compute_market_benchmark(args.data_start, args.data_end)
    df = pd.merge(quant_df, spy_df, on='Date', how='left', validate='many_to_one')
    df['target_alpha_1d'] = df['raw_target_1d'] - df['market_return_1d']
    df = df.dropna(subset=['target_alpha_1d']).reset_index(drop=True)

    print("\n[5/6] Merging trade-date LLM features...")
    df = merge_scoped_news_data(
        df,
        tickers_df,
        news_path=args.news_features,
        model_id=args.news_model_id,
        prompt_version=args.news_prompt_version,
    )

    # Only quantitative inputs are standardized. News judgments are already
    # semantically bounded, and a no-news zero must remain exactly zero.
    train_rows = df['Date'] < pd.Timestamp(args.train_end)
    scaler = StandardScaler()
    df.loc[train_rows, QUANT_FEATURE_NAMES] = scaler.fit_transform(
        df.loc[train_rows, QUANT_FEATURE_NAMES]
    )
    other_rows = ~train_rows
    df.loc[other_rows, QUANT_FEATURE_NAMES] = scaler.transform(
        df.loc[other_rows, QUANT_FEATURE_NAMES]
    )

    store = StockSequenceStore(
        df, QUANT_FEATURE_NAMES, SCOPED_LLM_FEATURE_NAMES
    )
    train_dataset = MultiStockDataset(
        store, end=args.train_end, seq_len=args.sequence_length
    )
    val_dataset = MultiStockDataset(
        store,
        start=args.train_end,
        end=args.validation_end,
        seq_len=args.sequence_length,
    )
    test_dataset = MultiStockDataset(
        store, start=args.validation_end, seq_len=args.sequence_length
    )
    if not train_dataset or not val_dataset or not test_dataset:
        raise ValueError("one or more chronological splits contain no samples")
    print(
        f"  Samples — train: {len(train_dataset):,}, "
        f"validation: {len(val_dataset):,}, test: {len(test_dataset):,}"
    )

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.evaluation_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.evaluation_batch_size, shuffle=False
    )

    use_news = args.model_mode == 'quant-news'
    print(
        f"\n[6/6] Training {'quant GRU + scoped news MLP' if use_news else 'quant-only GRU'}..."
    )
    model = TwoBranchModel(
        quant_dim=len(QUANT_FEATURE_NAMES),
        news_dim=len(SCOPED_LLM_FEATURE_NAMES),
        use_news=use_news,
    ).to(device)
    criterion = DirectionalBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        epoch_batches = len(train_loader)
        for batch_index, (quant_seq, news_today, batch_y, _) in enumerate(
            train_loader, start=1
        ):
            quant_seq = quant_seq.to(device)
            news_today = news_today.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(quant_seq, news_today)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            if (
                args.progress_every > 0
                and (
                    batch_index % args.progress_every == 0
                    or batch_index == epoch_batches
                )
            ):
                elapsed = time.time() - epoch_start
                batches_per_second = batch_index / max(elapsed, 1e-8)
                remaining_minutes = (
                    (epoch_batches - batch_index)
                    / max(batches_per_second, 1e-8)
                    / 60
                )
                print(
                    f"    Epoch {epoch + 1}: batch "
                    f"{batch_index:,}/{epoch_batches:,} | "
                    f"loss {running_loss / batch_index:.6f} | "
                    f"{batches_per_second:.1f} batches/s | "
                    f"ETA {remaining_minutes:.1f} min"
                )

        metrics = evaluate(model, val_loader)
        epoch_minutes = (time.time() - epoch_start) / 60
        print(
            f"  Epoch {epoch + 1}/{args.epochs} ({epoch_minutes:.1f} min) | "
            f"loss {running_loss / len(train_loader):.6f} | "
            f"val hit {metrics['hit_rate'] * 100:.2f}% | "
            f"daily IC {metrics['mean_daily_ic']:.4f} | "
            f"ICIR {metrics['icir']:.4f}"
        )
        val_ic = metrics['mean_daily_ic']
        if np.isfinite(val_ic) and val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"  Early stopping after {args.patience} epochs "
                    "without daily-IC improvement"
                )
                break

    if best_state is None:
        raise RuntimeError("validation daily IC was never finite")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    test_start_year = pd.Timestamp(args.validation_end).year
    test_end_year = (pd.Timestamp(args.data_end) - pd.Timedelta(days=1)).year
    test_period = (
        str(test_start_year)
        if test_start_year == test_end_year
        else f"{test_start_year}–{test_end_year}"
    )
    _print_metrics(
        f"OUT-OF-SAMPLE TEST ({test_period})",
        test_metrics,
    )

    if args.save_model:
        torch.save({
            'state_dict': model.state_dict(),
            'model_mode': args.model_mode,
            'quant_features': QUANT_FEATURE_NAMES,
            'news_features': SCOPED_LLM_FEATURE_NAMES,
            'quant_scaler_mean': scaler.mean_,
            'quant_scaler_scale': scaler.scale_,
            'sequence_length': args.sequence_length,
            'train_end': args.train_end,
            'validation_end': args.validation_end,
            'target': 'same-session open-to-close stock return minus SPY',
        }, args.save_model)
        print(f"Saved checkpoint to {args.save_model}")
    return model, test_metrics


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the leakage-safe multi-stock GRU/news model."
    )
    parser.add_argument('--tickers', default=TICKERS_CSV)
    parser.add_argument('--news-features')
    parser.add_argument('--news-model-id')
    parser.add_argument('--news-prompt-version')
    parser.add_argument('--price-cache')
    parser.add_argument(
        '--yfinance-cache',
        default=str(Path(SCRIPT_DIR) / '.yfinance_cache'),
    )
    parser.add_argument('--download-batch-size', type=int, default=25)
    parser.add_argument('--download-retries', type=int, default=3)
    parser.add_argument('--minimum-price-coverage', type=float, default=0.98)
    parser.add_argument('--data-start', default=DATA_START)
    parser.add_argument('--data-end', default=DATA_END)
    parser.add_argument('--train-end', default=TRAIN_END)
    parser.add_argument('--validation-end', default=VAL_END)
    parser.add_argument('--sequence-length', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--evaluation-batch-size', type=int, default=1024)
    parser.add_argument(
        '--progress-every',
        type=int,
        default=250,
        help="Print within-epoch progress every N training batches; 0 disables it.",
    )
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument(
        '--model-mode',
        choices=('quant-news', 'quant-only'),
        default='quant-news',
        help="Run quant-only separately as the required news ablation.",
    )
    parser.add_argument('--save-model')
    return parser


if __name__ == '__main__':
    train_model(build_parser().parse_args())
