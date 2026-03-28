"""
Мультипарный Live Trading — запускает бота на нескольких парах одновременно.

Запуск:
    python -m ai.multi_trader

Каждая пара работает в своём потоке с отдельным состоянием.
"""

import os
import sys
import json
import signal
import logging
import threading
from datetime import datetime, timezone

from ai.live_config import LiveConfig
from ai.live_trader import LiveGridTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/multi_trader.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("MultiTrader")

# Пары для торговли
DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MULTI_CONFIG_PATH = os.path.join(DATA_DIR, "multi_config.json")


def load_multi_config() -> dict:
    """Загружает конфигурацию мультипарного бота."""
    if os.path.exists(MULTI_CONFIG_PATH):
        with open(MULTI_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_multi_config(config: dict):
    """Сохраняет конфигурацию."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MULTI_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def run_trader(symbol: str, base_config: LiveConfig):
    """Запускает бота для одной пары (в потоке)."""
    try:
        # Копируем конфиг и меняем пару
        import copy
        config = copy.deepcopy(base_config)
        config.symbol = symbol

        logger.info("🚀 Запуск бота для %s", symbol)
        trader = LiveGridTrader(config)
        trader.start()
    except Exception as e:
        logger.error("💥 Бот %s завершился с ошибкой: %s", symbol, e)


def main():
    """Запуск мультипарного бота."""
    import argparse
    parser = argparse.ArgumentParser(description="Multi-pair Grid Trading Bot")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Список пар (например: BTC/USDT ETH/USDT)")
    parser.add_argument("--investment", type=float, default=None,
                        help="Депозит на каждую пару ($)")
    args = parser.parse_args()

    base_config = LiveConfig.load()

    # Определяем пары
    multi_conf = load_multi_config()
    symbols = args.symbols or multi_conf.get("symbols", DEFAULT_SYMBOLS)

    # Депозит на каждую пару
    per_pair_investment = args.investment or multi_conf.get(
        "per_pair_investment",
        base_config.investment / len(symbols)
    )
    base_config.investment = per_pair_investment

    if not base_config.api_key or not base_config.api_secret:
        print("❌ Установите переменные окружения:")
        print("   export BINANCE_API_KEY='ваш_ключ'")
        print("   export BINANCE_API_SECRET='ваш_секрет'")
        sys.exit(1)

    # Сохраняем конфиг
    save_multi_config({
        "symbols": symbols,
        "per_pair_investment": per_pair_investment,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info("=" * 60)
    logger.info("🚀 МУЛЬТИПАРНЫЙ БОТ")
    logger.info("   Пар: %d | Депозит на пару: $%.2f", len(symbols), per_pair_investment)
    logger.info("   Пары: %s", ", ".join(symbols))
    logger.info("   Testnet: %s | AI: %s", base_config.testnet, base_config.use_ai)
    logger.info("=" * 60)

    # Запускаем потоки
    threads = []
    for symbol in symbols:
        t = threading.Thread(
            target=run_trader,
            args=(symbol, base_config),
            name=f"bot-{symbol.replace('/', '-')}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info("✅ Поток запущен: %s", symbol)

    # Ждём завершения
    def handle_signal(sig, frame):
        logger.info("📛 Получен сигнал завершения...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Основной поток ждёт
    try:
        for t in threads:
            t.join()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🏁 Мультипарный бот завершён")


if __name__ == "__main__":
    main()
