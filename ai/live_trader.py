"""
Live Grid Trading Bot — реальная торговля на Binance.

Запуск:
    # Testnet (безопасно, фейковые деньги):
    export BINANCE_API_KEY="your_testnet_key"
    export BINANCE_API_SECRET="your_testnet_secret"
    python -m ai.live_trader

    # Реальная торговля:
    Измените testnet=False в data/live_config.json
"""

import os
import sys
import json
import time
import signal
import logging
import tempfile
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

from ai.live_config import LiveConfig
from ai.safety import SafetyGuard
from ai.feature_engineer import add_indicators, FEATURE_COLUMNS

logger = logging.getLogger("LiveTrader")


def setup_logging(symbol: str = "BTC_USDT"):
    """Настройка логирования с уникальным файлом для каждой пары."""
    safe = symbol.replace("/", "_")
    os.makedirs("data", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"data/live_trader_{safe}.log", encoding="utf-8"),
        ],
    )

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


class LiveGridTrader:
    """Live Grid + AI бот для Binance."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.running = False

        # Путь к файлу состояния
        safe_symbol = config.symbol.replace("/", "_")
        self.state_path = os.path.join(DATA_DIR, f"live_state_{safe_symbol}.json")
        self.command_path = os.path.join(DATA_DIR, "live_commands.json")

        # Инициализация биржи
        self.exchange = ccxt.binance({
            "apiKey": config.api_key,
            "secret": config.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if config.testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("🧪 TESTNET MODE — торговля без реальных денег")
        else:
            logger.warning("⚠️ РЕАЛЬНАЯ ТОРГОВЛЯ — будут использованы настоящие деньги!")

        # Безопасность
        self.safety = SafetyGuard(config.safety)

        # AI модель (опционально)
        self.model = None
        self.scaler = None
        if config.use_ai:
            try:
                from ai.model import load_model
                self.model, self.scaler = load_model(config.symbol)
                if self.model:
                    logger.info("🧠 AI модель загружена для %s", config.symbol)
                else:
                    logger.warning("AI модель не найдена — работаем без AI")
            except Exception as e:
                logger.warning("AI модель недоступна: %s", e)

        # Состояние
        self.state = self._default_state()

    def _default_state(self) -> dict:
        return {
            "version": 1,
            "symbol": self.config.symbol,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "balance": self.config.investment,
            "initial_investment": self.config.investment,
            "current_equity": self.config.investment,
            "grid": {
                "lower": 0, "upper": 0,
                "levels": [],
                "bought_levels": {},
                "order_size": self.config.investment / self.config.grid_count,
            },
            "ai_state": {
                "signal": 0.0,
                "prediction_history": [],
                "last_prediction_time": None,
                "accuracy": 0.5,
                "correct": 0, "total": 0,
            },
            "pending_orders": {},
            "trade_history": [],
            "equity_curve": [],
            "safety": {
                "peak_equity": self.config.investment,
                "current_drawdown": 0.0,
                "total_position_value": 0.0,
            },
            "error_log": [],
        }

    # ═══════════════════════════════════════════
    # Состояние — чтение/запись
    # ═══════════════════════════════════════════

    def load_state(self):
        """Загружает состояние с диска."""
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
            logger.info("📂 Состояние загружено (%d сделок в истории)",
                       len(self.state.get("trade_history", [])))
        else:
            logger.info("📂 Новый старт — создаём начальное состояние")

    def save_state(self):
        """Атомарная запись состояния (через temp файл)."""
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(DATA_DIR, exist_ok=True)

        # Атомарная запись: temp → rename
        fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp_path, self.state_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def check_commands(self):
        """Проверяет команды от UI (pause/resume/stop)."""
        if not os.path.exists(self.command_path):
            return

        try:
            with open(self.command_path) as f:
                commands = json.load(f)
            os.unlink(self.command_path)

            cmd = commands.get("command", "")
            if cmd == "pause":
                self.state["status"] = "paused"
                logger.info("⏸️ Торговля приостановлена (команда от UI)")
            elif cmd == "resume":
                self.state["status"] = "running"
                logger.info("▶️ Торговля возобновлена")
            elif cmd == "stop":
                self.state["status"] = "stopped"
                self.running = False
                logger.info("⏹️ Торговля остановлена")
        except Exception as e:
            logger.error("Ошибка чтения команд: %s", e)

    # ═══════════════════════════════════════════
    # Сетка
    # ═══════════════════════════════════════════

    def setup_grid(self, price: float):
        """Настраивает сетку уровней вокруг текущей цены."""
        rp = self.config.range_pct
        lower = price * (1 - rp / 100)
        upper = price * (1 + rp / 100)
        step = (upper - lower) / self.config.grid_count
        levels = [round(lower + i * step, 8) for i in range(1, self.config.grid_count)]

        self.state["grid"]["lower"] = lower
        self.state["grid"]["upper"] = upper
        self.state["grid"]["levels"] = levels

        logger.info("📐 Сетка: %.2f — %.2f (%d уровней, шаг %.2f)",
                    lower, upper, len(levels), step)

    # ═══════════════════════════════════════════
    # AI предсказания
    # ═══════════════════════════════════════════

    def update_ai_signal(self):
        """Обновляет AI сигнал на основе свежих данных."""
        if not self.model or not self.scaler:
            return

        try:
            from ai.data_collector import update_data
            exchange_public = ccxt.binance({"enableRateLimit": True})
            df = update_data(exchange_public, self.config.symbol, "1h")
            df = add_indicators(df).dropna().reset_index(drop=True)

            if len(df) < 48:
                return

            features = df[FEATURE_COLUMNS].iloc[-48:].values
            scaled = self.scaler.transform(features)
            X = np.expand_dims(scaled, axis=0)
            probs = self.model.predict(X, verbose=0)[0]

            raw_signal = float(probs[2] - probs[0])  # bullish - bearish

            # Скользящее среднее
            ai_state = self.state["ai_state"]
            ai_state["prediction_history"].append(raw_signal)
            if len(ai_state["prediction_history"]) > 5:
                ai_state["prediction_history"].pop(0)

            history = ai_state["prediction_history"]
            weights = list(range(1, len(history) + 1))
            ai_state["signal"] = sum(s * w for s, w in zip(history, weights)) / sum(weights)

            # Снижаем влияние при низкой точности
            acc = ai_state.get("accuracy", 0.5)
            if acc < 0.4:
                ai_state["signal"] *= 0.3
            elif acc < 0.5:
                ai_state["signal"] *= 0.6

            ai_state["last_prediction_time"] = datetime.now(timezone.utc).isoformat()

            directions = ["📉 Bearish", "↔️ Sideways", "📈 Bullish"]
            idx = int(np.argmax(probs))
            logger.info("🧠 AI: %s (signal=%.2f, conf=%.0f%%)",
                       directions[idx], ai_state["signal"], probs[idx] * 100)

        except Exception as e:
            logger.error("AI prediction error: %s", e)

    # ═══════════════════════════════════════════
    # Основной цикл торговли
    # ═══════════════════════════════════════════

    def run_cycle(self):
        """Один цикл: проверка рынка → принятие решений → ордера."""

        # 1. Получаем текущую цену
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            price = ticker["last"]
        except Exception as e:
            logger.error("Ошибка получения цены: %s", e)
            return

        grid = self.state["grid"]
        order_size = grid["order_size"]

        # 2. Проверяем ожидающие ордера
        self._check_pending_orders()

        # 3. AI предсказание (каждые N часов)
        ai_state = self.state["ai_state"]
        last_pred = ai_state.get("last_prediction_time")
        hours_since = 999
        if last_pred:
            try:
                dt = datetime.fromisoformat(last_pred)
                hours_since = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            except Exception:
                pass

        if hours_since >= self.config.ai_prediction_interval_hours:
            self.update_ai_signal()

        ai_signal = ai_state.get("signal", 0.0)

        # 4. Настройка сетки если ещё не настроена
        if not grid["levels"]:
            self.setup_grid(price)

        # 5. Проверяем диапазон
        if price < grid["lower"] or price > grid["upper"]:
            logger.info("⚡ Цена %.2f вышла за диапазон [%.2f, %.2f] — перенастройка",
                       price, grid["lower"], grid["upper"])
            self._cancel_all_orders()
            self.setup_grid(price)
            grid["bought_levels"] = {}
            return

        # 6. Торговая логика
        if self.state["status"] != "running":
            return

        balance = self.state["balance"]
        bought = grid["bought_levels"]

        for lvl in grid["levels"]:
            lvl_key = str(lvl)

            # ═══ BUY ═══
            if price <= lvl and lvl_key not in bought and balance >= order_size:
                # AI фильтр
                if ai_signal < -0.4:
                    continue

                # RSI фильтр (если есть данные)
                # ... можно добавить позже

                # Проверка безопасности
                amount = order_size / price
                allowed, reason = self.safety.check_can_buy(self.state, price, amount)
                if not allowed:
                    logger.info("🛡️ Покупка заблокирована: %s", reason)
                    continue

                # Размещаем ордер
                order = self._place_order("buy", price, amount, lvl)
                if order:
                    bought[lvl_key] = {
                        "amount": amount,
                        "buy_price": price,
                        "buy_time": datetime.now(timezone.utc).isoformat(),
                        "order_id": order.get("id", ""),
                    }
                    self.state["balance"] -= order_size
                    self._log_trade("buy", price, amount, order)

            # ═══ SELL ═══
            elif price >= lvl and lvl_key in bought:
                pos = bought[lvl_key]

                # AI фильтр: задержка продажи при сильном bullish
                if ai_signal > 0.5:
                    buy_time = pos.get("buy_time", "")
                    if buy_time:
                        try:
                            dt = datetime.fromisoformat(buy_time)
                            hold_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                            price_gain = (price - pos["buy_price"]) / pos["buy_price"]
                            if hold_hours < 24 and price_gain < 0.01:
                                continue  # Подождём
                        except Exception:
                            pass

                # Стоп-лосс проверка
                if self.safety.check_should_force_sell(pos, price):
                    logger.warning("🔴 СТОП-ЛОСС: принудительная продажа!")

                amount = pos["amount"]
                order = self._place_order("sell", price, amount, lvl)
                if order:
                    sell_value = amount * price
                    buy_cost = amount * pos["buy_price"]
                    fee = sell_value * self.config.fee_rate
                    profit = sell_value - buy_cost - fee * 2
                    self.state["balance"] += sell_value
                    del bought[lvl_key]
                    self._log_trade("sell", price, amount, order, profit)

        # Стоп-лосс для всех позиций
        for lvl_key in list(bought.keys()):
            pos = bought[lvl_key]
            if self.safety.check_should_force_sell(pos, price):
                amount = pos["amount"]
                order = self._place_order("sell", price, amount, float(lvl_key))
                if order:
                    sell_value = amount * price
                    buy_cost = amount * pos["buy_price"]
                    profit = sell_value - buy_cost - (sell_value + buy_cost) * self.config.fee_rate
                    self.state["balance"] += sell_value
                    del bought[lvl_key]
                    self._log_trade("sell", price, amount, order, profit)

        # 7. Обновляем equity
        holdings_value = sum(
            pos["amount"] * price
            for pos in bought.values()
        )
        equity = self.state["balance"] + holdings_value
        self.state["current_equity"] = equity

        # Обновляем безопасность
        safety = self.state["safety"]
        safety["total_position_value"] = holdings_value
        safety["peak_equity"] = max(safety.get("peak_equity", equity), equity)
        if safety["peak_equity"] > 0:
            safety["current_drawdown"] = (safety["peak_equity"] - equity) / safety["peak_equity"]

        # Equity curve (каждый час макс)
        curve = self.state["equity_curve"]
        if not curve or (len(curve) > 0 and
                        datetime.now(timezone.utc).isoformat()[:13] != curve[-1].get("t", "")[:13]):
            curve.append({
                "t": datetime.now(timezone.utc).isoformat(),
                "e": round(equity, 2),
                "p": round(price, 2),
            })
            # Хранить макс 720 точек (30 дней)
            if len(curve) > 720:
                curve.pop(0)

    # ═══════════════════════════════════════════
    # Ордера
    # ═══════════════════════════════════════════

    def _place_order(self, side: str, price: float, amount: float, level: float) -> dict:
        """Размещает лимитный ордер на бирже."""
        try:
            if side == "buy":
                order = self.exchange.create_limit_buy_order(
                    self.config.symbol, amount, price
                )
            else:
                order = self.exchange.create_limit_sell_order(
                    self.config.symbol, amount, price
                )

            logger.info("📋 %s ордер: %.6f %s @ %.2f (ID: %s)",
                       side.upper(), amount, self.config.symbol, price,
                       order.get("id", "?"))
            return order

        except ccxt.InsufficientFunds:
            logger.warning("💰 Недостаточно средств для %s %.6f @ %.2f", side, amount, price)
            return None
        except ccxt.InvalidOrder as e:
            logger.warning("❌ Невалидный ордер: %s", e)
            return None
        except Exception as e:
            logger.error("Ошибка ордера: %s", e)
            self._log_error(str(e))
            return None

    def _cancel_all_orders(self):
        """Отменяет все открытые ордера."""
        try:
            open_orders = self.exchange.fetch_open_orders(self.config.symbol)
            for order in open_orders:
                self.exchange.cancel_order(order["id"], self.config.symbol)
                logger.info("🚫 Отменён ордер %s", order["id"])
        except Exception as e:
            logger.error("Ошибка отмены ордеров: %s", e)

    def _check_pending_orders(self):
        """Проверяет статус ожидающих ордеров."""
        pending = self.state.get("pending_orders", {})
        for order_id in list(pending.keys()):
            try:
                order = self.exchange.fetch_order(order_id, self.config.symbol)
                if order["status"] == "closed":
                    del pending[order_id]
                    logger.info("✅ Ордер %s исполнен", order_id)
                elif order["status"] == "canceled":
                    del pending[order_id]
            except Exception:
                pass

    def _log_trade(self, side: str, price: float, amount: float,
                   order: dict, profit: float = 0.0):
        """Записывает сделку в историю."""
        self.state["trade_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "price": round(price, 2),
            "amount": round(amount, 8),
            "fee": round(price * amount * self.config.fee_rate, 4),
            "profit": round(profit, 4),
            "order_id": order.get("id", ""),
        })

        emoji = "🟢" if side == "buy" else ("🟩" if profit > 0 else "🟥")
        logger.info("%s %s %.6f @ %.2f | P&L: $%.2f",
                   emoji, side.upper(), amount, price, profit)

    def _log_error(self, message: str):
        """Записывает ошибку в состояние."""
        errors = self.state.get("error_log", [])
        errors.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })
        # Хранить макс 100 ошибок
        if len(errors) > 100:
            errors.pop(0)
        self.state["error_log"] = errors

    # ═══════════════════════════════════════════
    # Запуск
    # ═══════════════════════════════════════════

    def reconcile(self):
        """Сверка состояния с биржей после перезапуска."""
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {}).get("free", 0)
            logger.info("💰 Баланс на бирже: %.2f USDT", usdt)

            open_orders = self.exchange.fetch_open_orders(self.config.symbol)
            logger.info("📋 Открытых ордеров: %d", len(open_orders))
        except Exception as e:
            logger.error("Ошибка сверки: %s", e)

    def start(self):
        """Основной цикл бота."""
        self.running = True
        os.makedirs(DATA_DIR, exist_ok=True)

        # Загружаем сохранённое состояние
        self.load_state()
        self.state["status"] = "running"

        # Сверка с биржей
        self.reconcile()

        logger.info("=" * 50)
        logger.info("🚀 СТАРТ: %s | Депозит: $%.2f | Сетка: %d уровней, %.1f%%",
                    self.config.symbol, self.config.investment,
                    self.config.grid_count, self.config.range_pct)
        logger.info("   Testnet: %s | AI: %s | Интервал: %dс",
                    self.config.testnet, bool(self.model),
                    self.config.poll_interval_seconds)
        logger.info("=" * 50)

        # Обработка сигналов завершения
        def handle_signal(sig, frame):
            logger.info("📛 Получен сигнал завершения — сохраняю состояние...")
            self.state["status"] = "stopped"
            self.save_state()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Основной цикл
        retry_delay = 1
        while self.running:
            try:
                # Проверяем команды от UI
                self.check_commands()

                if not self.running:
                    break

                # Проверка безопасности
                action = self.safety.pre_cycle_check(self.state)
                if action == "pause":
                    self.state["status"] = "paused"
                    logger.warning("⏸️ Торговля приостановлена (безопасность)")
                elif action == "stop":
                    break

                # Торговый цикл
                if self.state["status"] == "running":
                    self.run_cycle()

                # Сохраняем состояние
                self.save_state()

                # Сброс задержки при успехе
                retry_delay = 1

                # Ждём
                time.sleep(self.config.poll_interval_seconds)

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                logger.warning("🌐 Сетевая ошибка: %s (повтор через %dс)", e, retry_delay)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

            except ccxt.ExchangeError as e:
                logger.error("❌ Ошибка биржи: %s", e)
                self._log_error(str(e))
                time.sleep(10)

            except Exception as e:
                logger.critical("💥 Критическая ошибка: %s", e, exc_info=True)
                self._log_error(str(e))
                self.state["status"] = "error"
                self.save_state()
                break

        logger.info("🏁 Бот остановлен. Финальный баланс: $%.2f",
                    self.state.get("current_equity", 0))
        self.save_state()


def main():
    """Точка входа."""
    import argparse
    parser = argparse.ArgumentParser(description="Live Grid Trading Bot")
    parser.add_argument("--config", type=str, default=None,
                        help="Путь к файлу конфигурации (по умолчанию data/live_config.json)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Торговая пара (например ETH/USDT)")
    args = parser.parse_args()

    config = LiveConfig.load(args.config)

    if args.symbol:
        config.symbol = args.symbol

    setup_logging(config.symbol)

    if not config.api_key or not config.api_secret:
        print("❌ Установите переменные окружения:")
        print("   export BINANCE_API_KEY='ваш_ключ'")
        print("   export BINANCE_API_SECRET='ваш_секрет'")
        print()
        print("Для Testnet ключи можно получить на:")
        print("   https://testnet.binance.vision/")
        sys.exit(1)

    trader = LiveGridTrader(config)
    trader.start()


if __name__ == "__main__":
    main()
