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

# Пары и веса портфеля (больше вес = больше депозит)
# Топ-монеты получают больше, рисковые — меньше
DEFAULT_PORTFOLIO = {
    "BTC/USDT":  30,   # 30% — основа портфеля
    "ETH/USDT":  25,   # 25% — вторая по капитализации
    "SOL/USDT":  12,   # 12% — быстрорастущий L1
    "BNB/USDT":  12,   # 12% — экосистема Binance
    "LINK/USDT": 10,   # 10% — лидер оракулов
    "ADA/USDT":   6,   #  6% — средний риск
    "DOGE/USDT":  5,   #  5% — высокий риск, мем
}

DEFAULT_SYMBOLS = list(DEFAULT_PORTFOLIO.keys())

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


def run_trader(symbol: str, base_config: LiveConfig, investment: float = None):
    """Запускает бота для одной пары (в потоке)."""
    try:
        import copy
        config = copy.deepcopy(base_config)
        config.symbol = symbol
        if investment:
            config.investment = investment

        logger.info("🚀 Запуск бота для %s (депозит: $%.2f)", symbol, config.investment)
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
    total_investment = base_config.investment

    # Определяем пары и веса
    multi_conf = load_multi_config()
    portfolio = multi_conf.get("portfolio", DEFAULT_PORTFOLIO)
    symbols = args.symbols or list(portfolio.keys())

    # Рассчитываем депозит для каждой пары по весам
    total_weight = sum(portfolio.get(s, 10) for s in symbols)
    allocations = {}
    for symbol in symbols:
        weight = portfolio.get(symbol, 10)
        allocations[symbol] = round(total_investment * weight / total_weight, 2)

    if not base_config.api_key or not base_config.api_secret:
        print("❌ Установите переменные окружения:")
        print("   export BINANCE_API_KEY='ваш_ключ'")
        print("   export BINANCE_API_SECRET='ваш_секрет'")
        sys.exit(1)

    # Сохраняем конфиг
    save_multi_config({
        "portfolio": portfolio,
        "total_investment": total_investment,
        "allocations": allocations,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info("=" * 60)
    logger.info("🚀 МУЛЬТИПАРНЫЙ БОТ")
    logger.info("   Общий депозит: $%.2f | Пар: %d", total_investment, len(symbols))
    for sym, alloc in allocations.items():
        weight = portfolio.get(sym, 10)
        logger.info("   %s: $%.2f (%d%%)", sym, alloc, weight)
    logger.info("   Testnet: %s | AI: %s", base_config.testnet, base_config.use_ai)
    logger.info("=" * 60)

    # Запускаем потоки
    threads = []
    for symbol in symbols:
        investment = allocations[symbol]
        t = threading.Thread(
            target=run_trader,
            args=(symbol, base_config, investment),
            name=f"bot-{symbol.replace('/', '-')}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        logger.info("✅ Поток запущен: %s ($%.2f)", symbol, investment)

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
