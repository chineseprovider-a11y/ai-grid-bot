"""Сбор и хранение исторических данных OHLCV."""

import os
import time
import pandas as pd
import ccxt
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fetch_historical_data(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str = "1h",
    days: int = 90,
) -> pd.DataFrame:
    """Загружает исторические свечи с Binance (публичный API, без ключей)."""
    all_candles = []
    limit = 1000
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1  # следующая свеча после последней
        if len(candles) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def save_to_csv(df: pd.DataFrame, symbol: str, timeframe: str = "1h"):
    """Сохраняет данные в CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


def load_from_csv(symbol: str, timeframe: str = "1h"):
    """Загружает данные из CSV."""
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def update_data(exchange: ccxt.binance, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    """Обновляет кэшированные данные новыми свечами."""
    existing = load_from_csv(symbol, timeframe)
    if existing is not None and len(existing) > 0:
        last_ts = int(existing["timestamp"].iloc[-1].timestamp() * 1000) + 1
        new_candles = exchange.fetch_ohlcv(symbol, timeframe, since=last_ts, limit=1000)
        if new_candles:
            new_df = pd.DataFrame(new_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
            df = pd.concat([existing, new_df]).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        else:
            df = existing
    else:
        df = fetch_historical_data(exchange, symbol, timeframe)

    save_to_csv(df, symbol, timeframe)
    return df
