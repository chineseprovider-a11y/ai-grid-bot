"""Скрипт обучения AI-модели для всех торговых пар."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ccxt
from ai.data_collector import fetch_historical_data, save_to_csv
from ai.feature_engineer import prepare_training_data, FEATURE_COLUMNS
from ai.model import build_lstm_model, train_model, save_model

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]
SEQ_LENGTH = 48
HORIZON = 6
EPOCHS = 15


def train_all():
    exchange = ccxt.binance({"enableRateLimit": True})

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  Обучение модели: {symbol}")
        print(f"{'='*60}")

        # 1. Загрузка данных (90 дней, 1h свечи)
        print(f"  Загружаю данные за 90 дней...")
        df = fetch_historical_data(exchange, symbol, timeframe="1h", days=90)
        save_to_csv(df, symbol, "1h")
        print(f"  Загружено {len(df)} свечей")

        if len(df) < 200:
            print(f"  ПРОПУЩЕНО: недостаточно данных ({len(df)} < 200)")
            continue

        # 2. Подготовка данных
        print(f"  Подготовка признаков и sequences...")
        X_train, y_train, X_val, y_val, scaler = prepare_training_data(
            df, seq_length=SEQ_LENGTH, horizon=HORIZON
        )
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"  Features: {len(FEATURE_COLUMNS)}")

        # Баланс классов
        for cls in range(3):
            train_pct = (y_train == cls).sum() / len(y_train) * 100
            val_pct = (y_val == cls).sum() / len(y_val) * 100
            labels = ["bearish", "sideways", "bullish"]
            print(f"    {labels[cls]}: train={train_pct:.1f}%, val={val_pct:.1f}%")

        # 3. Обучение LSTM
        print(f"\n  Обучение LSTM ({EPOCHS} эпох, early stopping)...")
        model = build_lstm_model(SEQ_LENGTH, len(FEATURE_COLUMNS), n_classes=3)
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS)

        # 4. Результат
        val_acc = max(history.history["val_accuracy"])
        print(f"\n  Лучшая val_accuracy: {val_acc:.4f}")

        # 5. Сохранение
        save_model(model, scaler, symbol)
        print(f"  Модель сохранена: models/{symbol.replace('/', '_')}_lstm.keras")

    print(f"\n{'='*60}")
    print("  Все модели обучены!")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_all()
