"""LSTM модель для предсказания направления цены."""

import os
import numpy as np
import joblib
import logging

logger = logging.getLogger("AIModel")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_lstm_model(seq_length: int, n_features: int, n_classes: int = 3):
    """Создаёт улучшенную LSTM модель с регуляризацией."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        # LSTM слой 1 с L2 регуляризацией
        keras.layers.LSTM(
            48, return_sequences=True,
            input_shape=(seq_length, n_features),
            kernel_regularizer=keras.regularizers.l2(0.001),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # LSTM слой 2
        keras.layers.LSTM(
            24, return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.001),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # Dense слои
        keras.layers.Dense(16, activation="relu",
                          kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32):
    """Обучает модель с ранней остановкой и class weights."""
    from tensorflow import keras

    # Автоматический подсчёт весов классов (борьба с дисбалансом)
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(cls): total / (len(unique) * count)
                    for cls, count in zip(unique, counts)}
    logger.info("Class weights: %s", class_weight)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    return history


def save_model(model, scaler, symbol: str):
    """Сохраняет модель и scaler."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    model.save(os.path.join(MODEL_DIR, f"{safe_symbol}_lstm.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{safe_symbol}_scaler.pkl"))
    logger.info("Model saved for %s", symbol)


def load_model(symbol: str):
    """Загружает модель и scaler."""
    from tensorflow import keras

    safe_symbol = symbol.replace("/", "_")
    model_path = os.path.join(MODEL_DIR, f"{safe_symbol}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{safe_symbol}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, recent_features: np.ndarray) -> dict:
    """
    Предсказание на основе последних данных.
    recent_features: shape (seq_length, n_features) — необработанные признаки.
    Возвращает dict с предсказанием.
    """
    scaled = scaler.transform(recent_features)
    X = np.expand_dims(scaled, axis=0)  # (1, seq_length, n_features)

    probs = model.predict(X, verbose=0)[0]
    direction_idx = int(np.argmax(probs))
    directions = ["bearish", "sideways", "bullish"]

    return {
        "direction": directions[direction_idx],
        "confidence": float(probs[direction_idx]),
        "probabilities": {
            "bearish": float(probs[0]),
            "sideways": float(probs[1]),
            "bullish": float(probs[2]),
        },
    }
