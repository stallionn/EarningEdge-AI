"""
EarningsEdge AI — Phase 1.3
Price Data Collection via Alpaca Markets API

Pulls daily OHLCV for each ticker in data/raw/, ±10 days around earnings call date.
Computes 1-day, 3-day, 5-day post-call price returns.

Outputs:
    data/prices/{ticker}.parquet — full OHLCV + returns
    data/prices/returns_index.parquet — summary: ticker, call_date, ret_1d, ret_3d, ret_5d

Usage:
    python scripts/price_fetcher.py
    python scripts/price_fetcher.py --ticker AAPL MSFT NVDA  (specific tickers only)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, PRICES_DIR, RETURN_WINDOWS,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, RANDOM_SEED
)

np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("price_fetcher")

PRICES_DIR.mkdir(parents=True, exist_ok=True)


# ─── Alpaca Client ────────────────────────────────────────────────────────────
def get_alpaca_client():
    """Initialize Alpaca REST client."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        return StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except ImportError:
        log.error("alpaca-py not installed. Run: pip install alpaca-py")
        return None


# ─── Fetch OHLCV ──────────────────────────────────────────────────────────────
def fetch_ticker_ohlcv(
    ticker: str,
    call_dates: list[str],
    window_days: int = 10
) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV for a ticker covering all earnings call dates.
    Fetches one big range covering all call dates ± window_days.
    Returns DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        return _fallback_yfinance(ticker, call_dates, window_days)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        log.warning("Alpaca API keys not configured. Falling back to yfinance.")
        return _fallback_yfinance(ticker, call_dates, window_days)

    # Compute date range from all call dates
    parsed_dates = sorted([datetime.strptime(d, "%Y%m%d") for d in call_dates if _valid_date(d)])
    if not parsed_dates:
        return None

    start_dt = parsed_dates[0] - timedelta(days=window_days + 5)
    end_dt = parsed_dates[-1] + timedelta(days=window_days + 5)

    try:
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
        )
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            log.warning(f"No price data for {ticker}")
            return None

        # Reset multi-index if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        else:
            df = df.reset_index()

        # Standardize columns
        df = df.rename(columns={
            "timestamp": "date", "t": "date",
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap"
        })

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = ticker
        df = df.sort_values("date").reset_index(drop=True)

        log.info(f"  {ticker}: {len(df)} trading days fetched")
        return df

    except Exception as e:
        log.warning(f"Alpaca fetch failed for {ticker}: {e}. Trying yfinance fallback.")
        return _fallback_yfinance(ticker, call_dates, window_days)


def _fallback_yfinance(
    ticker: str, call_dates: list[str], window_days: int
) -> pd.DataFrame | None:
    """
    Fallback: use yfinance if Alpaca fails or keys not configured.
    yfinance is entirely free and needs no API key.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        return None

    parsed_dates = sorted([datetime.strptime(d, "%Y%m%d") for d in call_dates if _valid_date(d)])
    if not parsed_dates:
        return None

    start_dt = parsed_dates[0] - timedelta(days=window_days + 5)
    end_dt = parsed_dates[-1] + timedelta(days=window_days + 5)

    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(start=start_dt, end=end_dt, auto_adjust=True)
        if df.empty:
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["ticker"] = ticker
        df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        log.info(f"  {ticker}: {len(df)} trading days from yfinance")
        return df
    except Exception as e:
        log.error(f"yfinance also failed for {ticker}: {e}")
        return None


# ─── Compute Returns ──────────────────────────────────────────────────────────
def compute_post_call_returns(
    price_df: pd.DataFrame, call_date_str: str
) -> dict | None:
    """
    Compute 1-day, 3-day, 5-day returns after an earnings call.
    Uses next trading day open as entry (avoids earnings-day gap risk).
    Returns dict or None if data insufficient.
    """
    try:
        call_date = datetime.strptime(call_date_str, "%Y%m%d").date()
    except ValueError:
        return None

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    price_df = price_df.sort_values("date").reset_index(drop=True)

    # Find idx of first trading day >= call_date
    future_rows = price_df[price_df["date"] >= call_date]
    if future_rows.empty:
        return None

    entry_idx = future_rows.index[0]
    results = {"call_date": call_date_str}

    for window in RETURN_WINDOWS:
        exit_idx = entry_idx + window
        if exit_idx >= len(price_df):
            results[f"ret_{window}d"] = np.nan
            continue
        entry_close = price_df.at[entry_idx, "close"]
        exit_close = price_df.at[exit_idx, "close"]
        if entry_close == 0:
            results[f"ret_{window}d"] = np.nan
        else:
            results[f"ret_{window}d"] = (exit_close - entry_close) / entry_close

    return results


# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def get_all_ticker_call_dates() -> dict[str, list[str]]:
    """Scan data/raw/ and build {ticker: [call_date, ...]} mapping."""
    ticker_dates: dict[str, list[str]] = {}
    for fpath in RAW_DIR.glob("*.json"):
        try:
            with open(fpath, encoding="utf-8") as f:
                rec = json.load(f)
            ticker = rec.get("ticker", "").strip().upper()
            date = rec.get("date", "")
            if ticker and date and date != "19000101" and _valid_date(date):
                ticker_dates.setdefault(ticker, [])
                if date not in ticker_dates[ticker]:
                    ticker_dates[ticker].append(date)
        except Exception:
            continue
    return ticker_dates


def _valid_date(date_str: str) -> bool:
    """Check if date string is a valid YYYYMMDD."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d").date()
        return datetime(2015, 1, 1).date() <= d <= datetime.today().date()
    except ValueError:
        return False


def run_price_pipeline(target_tickers: list[str] | None = None):
    """Main entry point: fetch prices and compute returns for all tickers."""
    ticker_dates = get_all_ticker_call_dates()
    log.info(f"Found {len(ticker_dates)} unique tickers in raw data")

    if target_tickers:
        ticker_dates = {k: v for k, v in ticker_dates.items() if k in target_tickers}
        log.info(f"Filtered to {len(ticker_dates)} target tickers")

    all_return_records = []
    failed_tickers = []
    processed = 0

    for ticker, call_dates in ticker_dates.items():
        log.info(f"Processing {ticker} ({len(call_dates)} call dates)...")

        price_df = fetch_ticker_ohlcv(ticker, call_dates)
        if price_df is None or price_df.empty:
            log.warning(f"  No price data for {ticker}")
            failed_tickers.append(ticker)
            # Respect rate limits
            time.sleep(0.5)
            continue

        # Save full OHLCV
        parquet_path = PRICES_DIR / f"{ticker}.parquet"
        price_df.to_parquet(parquet_path, index=False)

        # Compute returns per call date
        for call_date in call_dates:
            ret = compute_post_call_returns(price_df, call_date)
            if ret:
                ret["ticker"] = ticker
                all_return_records.append(ret)

        processed += 1
        log.info(f"  ✓ {ticker}: saved {len(price_df)} price rows, {len(call_dates)} return records")

        # Rate limiting
        time.sleep(0.3)

        if processed % 50 == 0:
            log.info(f"Progress: {processed}/{len(ticker_dates)} tickers done")

    # Save returns index
    if all_return_records:
        returns_df = pd.DataFrame(all_return_records)
        returns_df = returns_df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
        idx_path = PRICES_DIR / "returns_index.parquet"
        returns_df.to_parquet(idx_path, index=False)

        log.info("=" * 60)
        log.info("PRICE FETCH SUMMARY")
        log.info("=" * 60)
        log.info(f"  Tickers processed:     {processed}")
        log.info(f"  Return records:        {len(returns_df)}")
        log.info(f"  Failed tickers:        {len(failed_tickers)}")
        if failed_tickers:
            log.info(f"  Failed list:           {failed_tickers[:10]}")
        log.info(f"  Returns index saved:   {idx_path}")
        log.info(f"  ret_1d null %:         {returns_df['ret_1d'].isna().mean():.1%}")
        log.info(f"  ret_3d null %:         {returns_df['ret_3d'].isna().mean():.1%}")
        log.info(f"  ret_5d null %:         {returns_df['ret_5d'].isna().mean():.1%}")

        if len(returns_df) >= 500:
            log.info("✓ Phase 1.3 checkpoint: ≥500 return records ready.")
        else:
            log.warning(f"⚠ Only {len(returns_df)} return records. Need ≥500.")
    else:
        log.error("No return records generated. Check your API credentials and raw data.")

    return all_return_records


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EarningsEdge Price Data Fetcher")
    parser.add_argument(
        "--ticker", nargs="+", default=None,
        help="Specific tickers to fetch (default: all from data/raw/)"
    )
    args = parser.parse_args()

    # Check if yfinance is available as fallback
    try:
        import yfinance
        log.info("yfinance available as Alpaca fallback ✓")
    except ImportError:
        log.info("Tip: install yfinance as Alpaca fallback: pip install yfinance")

    run_price_pipeline(target_tickers=args.ticker)


if __name__ == "__main__":
    main()
