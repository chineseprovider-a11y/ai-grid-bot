"""Инженерия признаков: технические индикаторы и подготовка данных для LSTM."""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы к OHLCV данным."""
    df = df.copy()

    # RSI (14)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR (14) — ключевой для ширины сетки
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    # EMA 20 и 50
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_cross"] = (df["ema_20"] - df["ema_50"]) / df["close"]

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volume SMA
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)

    # Price change features
    df["returns_1"] = df["close"].pct_change(1)
    df["returns_6"] = df["close"].pct_change(6)
    df["returns_24"] = df["close"].pct_change(24)

    return df


FEATURE_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position", "atr_pct",
    "ema_cross", "stoch_k", "stoch_d",
    "volume_ratio", "returns_1", "returns_6", "returns_24",
]


def create_target(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.005) -> pd.Series:
    """
    Создаёт целевую переменную: направление цены через `horizon` свечей.
    0 = bearish (< -threshold)
    1 = sideways (-threshold..+threshold)
    2 = bullish (> +threshold)
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    target = pd.Series(1, index=df.index)  # sideways by default
    target[future_return > threshold] = 2   # bullish
    target[future_return < -threshold] = 0  # bearish
    return target


def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int = 48):
    """Создаёт скользящие окна для LSTM."""
    X, y = [], []
    for i in range(seq_length, len(features) - 1):
        X.append(features[i - seq_length:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def prepare_training_data(df: pd.DataFrame, seq_length: int = 48, horizon: int = 6):
    """
    Полный пайплайн: индикаторы → признаки → target → sequences.
    Возвращает (X_train, y_train, X_val, y_val, scaler).
    """
    df = add_indicators(df)
    df["target"] = create_target(df, horizon=horizon)

    # Удаляем NaN (от индикаторов и от target)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)

    features = df[FEATURE_COLUMNS].values
    targets = df["target"].values.astype(int)

    # Масштабирование
    scaler = MinMaxScaler()
    split = int(len(features) * 0.8)
    scaler.fit(features[:split])
    features_scaled = scaler.transform(features)

    # Sequences
    X, y = create_sequences(features_scaled, targets, seq_length)

    # Train/val split (по времени, без перемешивания)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_val, y_val, scaler
