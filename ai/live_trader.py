"""
Live Grid Trading Bot — paper trading и реальная торговля на Binance.

Режимы:
    paper_trading=True  — реальные цены, симуляция ордеров (рекомендуется)
    testnet=True        — Binance Testnet (фейковые цены)
    оба False           — реальная торговля

Запуск:
    python -m ai.live_trader
    python -m ai.multi_trader
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

        # ═══ Paper trading: реальные цены без ордеров ═══
        self.paper_mode = config.paper_trading

        if self.paper_mode:
            # Публичная биржа (без ключей) — только для получения цен
            self.exchange = ccxt.binance({"enableRateLimit": True})
            logger.info("📝 PAPER TRADING — реальные цены, симуляция ордеров")
        else:
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

        # Загружаем информацию о рынках (для округления)
        self.market_info = {}
        try:
            self.exchange.load_markets()
            if config.symbol in self.exchange.markets:
                m = self.exchange.markets[config.symbol]
                self.market_info = {
                    "amount_precision": m.get("precision", {}).get("amount", 8),
                    "price_precision": m.get("precision", {}).get("price", 2),
                    "min_amount": m.get("limits", {}).get("amount", {}).get("min", 0),
                    "min_cost": m.get("limits", {}).get("cost", {}).get("min", 0),
                }
                logger.info("📊 %s: min_cost=$%s", config.symbol, self.market_info["min_cost"])
        except Exception as e:
            logger.warning("Не удалось загрузить рынки: %s", e)

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

        # ═══ Кеш быстрых индикаторов ═══
        self._indicator_cache = {
            "rsi": 50.0,
            "ema_fast": 0.0,   # EMA 12
            "ema_slow": 0.0,   # EMA 26
            "trend": "neutral", # "up", "down", "neutral"
            "last_update": None,
        }

        # Cooldown после стоп-лосса (не покупать N часов)
        self._last_stop_loss_time = None
        self._stop_loss_cooldown_hours = 3

        # Защита от спама AI-решений
        self._last_logged_decision = None

        # Состояние
        self.state = self._default_state()

    def _default_state(self) -> dict:
        return {
            "version": 2,
            "symbol": self.config.symbol,
            "mode": "paper" if self.paper_mode else ("testnet" if self.config.testnet else "live"),
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
                "grid_range_pct": self.config.range_pct,
                "grid_shift": 0.0,
                "size_multiplier": 1.0,
                "take_profit_pct": 0.0,
            },
            "indicators": {
                "rsi": 50.0,
                "ema_fast": 0.0,
                "ema_slow": 0.0,
                "trend": "neutral",
            },
            "ai_decisions": [],  # Лог AI-решений для дашборда
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
            # Миграция: добавляем новые поля если их нет
            self.state.setdefault("ai_decisions", [])
            self.state.setdefault("indicators", {"rsi": 50, "ema_fast": 0, "ema_slow": 0, "trend": "neutral"})
            self.state.setdefault("mode", "paper" if self.paper_mode else "testnet")
            logger.info("📂 Состояние загружено (%d сделок в истории)",
                       len(self.state.get("trade_history", [])))
        else:
            logger.info("📂 Новый старт — создаём начальное состояние")

    def save_state(self):
        """Атомарная запись состояния (через temp файл)."""
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(DATA_DIR, exist_ok=True)

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
    # Быстрые индикаторы (каждый цикл)
    # ═══════════════════════════════════════════

    def update_fast_indicators(self):
        """
        Обновляет RSI и EMA каждый цикл на основе свежих OHLCV данных.
        Используется для trend filter и дополнительной фильтрации.
        """
        try:
            # Получаем последние 30 свечей (1h) — достаточно для RSI-14 и EMA-26
            ohlcv = self.exchange.fetch_ohlcv(self.config.symbol, "1h", limit=30)
            if len(ohlcv) < 26:
                return

            closes = np.array([c[4] for c in ohlcv])

            # RSI-14
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            # EMA-12 и EMA-26
            def ema(data, period):
                multiplier = 2 / (period + 1)
                result = data[0]
                for val in data[1:]:
                    result = (val - result) * multiplier + result
                return result

            ema_fast = ema(closes, 12)
            ema_slow = ema(closes, 26)

            # Определяем тренд
            if ema_fast > ema_slow * 1.002:  # 0.2% запас от шума
                trend = "up"
            elif ema_fast < ema_slow * 0.998:
                trend = "down"
            else:
                trend = "neutral"

            self._indicator_cache = {
                "rsi": round(rsi, 1),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "trend": trend,
                "last_update": datetime.now(timezone.utc).isoformat(),
            }

            # Сохраняем в state для дашборда
            self.state["indicators"] = {
                "rsi": self._indicator_cache["rsi"],
                "ema_fast": self._indicator_cache["ema_fast"],
                "ema_slow": self._indicator_cache["ema_slow"],
                "trend": trend,
            }

        except Exception as e:
            logger.debug("Ошибка обновления индикаторов: %s", e)

    # ═══════════════════════════════════════════
    # Сетка
    # ═══════════════════════════════════════════

    def setup_grid(self, price: float, ai_signal: float = 0.0, keep_positions: bool = False):
        """
        Настраивает сетку уровней вокруг текущей цены.
        AI влияет на диапазон и смещение.
        keep_positions=True: при перестройке сохраняет купленные позиции.
        """
        rp = self.config.range_pct

        # ═══ AI: Динамический диапазон ═══
        if self.model and abs(ai_signal) > 0.1:
            if ai_signal > 0.3:
                rp *= (1 - ai_signal * 0.2)  # до -20%
            elif ai_signal < -0.3:
                rp *= (1 + abs(ai_signal) * 0.3)  # до +30%
            rp = max(2.0, min(rp, 10.0))

        # ═══ AI: Смещение центра сетки ═══
        shift = 0.0
        if self.model and abs(ai_signal) > 0.2:
            shift = ai_signal * 0.2

        half_range = price * rp / 100
        center = price * (1 + shift * rp / 100)
        lower = center - half_range
        upper = center + half_range

        # Расширяем чтобы покрыть открытые позиции
        if keep_positions:
            bought = self.state["grid"].get("bought_levels", {})
            for pos in bought.values():
                bp = pos.get("buy_price", 0)
                if bp > 0:
                    lower = min(lower, bp * 0.98)
                    upper = max(upper, bp * 1.03)

        step = (upper - lower) / self.config.grid_count
        levels = [round(lower + i * step, 8) for i in range(1, self.config.grid_count)]

        self.state["grid"]["lower"] = lower
        self.state["grid"]["upper"] = upper
        self.state["grid"]["levels"] = levels

        ai_info = ""
        if self.model:
            ai_info = f" | AI: range={rp:.1f}%, shift={shift:+.1f}"
        logger.info("📐 Сетка: %.2f — %.2f (%d уровней, шаг %.2f%s)",
                    lower, upper, len(levels), step, ai_info)

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

            raw_signal = float(probs[2] - probs[0])

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

            # Записываем AI-параметры
            sig = ai_state["signal"]
            ai_state["size_multiplier"] = round(max(0.6, min(1.0 + sig * 0.5, 1.5)), 2)
            ai_state["take_profit_pct"] = round(sig * 1.5 if sig > 0.3 else (sig * 0.3 if sig < -0.3 else 0.0), 2)

            directions = ["📉 Bearish", "↔️ Sideways", "📈 Bullish"]
            idx = int(np.argmax(probs))
            logger.info("🧠 AI: %s (signal=%.2f, conf=%.0f%%, size=x%.1f, tp=%.1f%%)",
                       directions[idx], ai_state["signal"], probs[idx] * 100,
                       ai_state["size_multiplier"], ai_state["take_profit_pct"])

        except Exception as e:
            logger.error("AI prediction error: %s", e)

    def _log_ai_decision(self, action: str, reason: str, price: float, ai_signal: float):
        """Записывает AI-решение в лог для дашборда. Фильтрует спам."""
        # Антиспам: не логируем одинаковые блокировки чаще чем раз в 5 минут
        decision_key = f"{action}:{reason.split(',')[0].split('(')[0].strip()}"
        now = datetime.now(timezone.utc)
        if action in ("block_buy", "skip_upper", "hold"):
            if self._last_logged_decision:
                last_key, last_time = self._last_logged_decision
                if last_key == decision_key:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < 300:  # 5 минут
                        return
            self._last_logged_decision = (decision_key, now)

        decisions = self.state.get("ai_decisions", [])
        decisions.append({
            "t": now.isoformat(),
            "action": action,
            "reason": reason,
            "price": round(price, 2),
            "signal": round(ai_signal, 3),
            "rsi": self._indicator_cache.get("rsi", 50),
            "trend": self._indicator_cache.get("trend", "neutral"),
        })
        # Храним последние 200 решений
        if len(decisions) > 200:
            decisions.pop(0)
        self.state["ai_decisions"] = decisions

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

        # 2. Проверяем ожидающие ордера (не в paper mode)
        if not self.paper_mode:
            self._check_pending_orders()

        # 3. Обновляем быстрые индикаторы (RSI, EMA, тренд)
        self.update_fast_indicators()
        trend = self._indicator_cache.get("trend", "neutral")
        rsi = self._indicator_cache.get("rsi", 50)

        # 4. AI предсказание (каждые N часов)
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

        # 5. Настройка сетки если ещё не настроена
        if not grid["levels"]:
            self.setup_grid(price, ai_signal)

        # 6. Проверяем диапазон — расширяем сетку, НЕ теряем позиции
        if price < grid["lower"] or price > grid["upper"]:
            logger.info("⚡ Цена %.2f вышла за диапазон [%.2f, %.2f] — расширяем сетку",
                       price, grid["lower"], grid["upper"])
            if not self.paper_mode:
                self._cancel_all_orders()
            self.setup_grid(price, ai_signal, keep_positions=True)

        # ═══ AI: Адаптивный размер ордера ═══
        base_order_size = grid["order_size"]
        if self.model and abs(ai_signal) > 0.2:
            size_multiplier = 1.0 + ai_signal * 0.5
            size_multiplier = max(0.6, min(size_multiplier, 1.5))
            adjusted_order_size = base_order_size * size_multiplier
        else:
            adjusted_order_size = base_order_size
            size_multiplier = 1.0

        # ═══ AI: Динамический take-profit ═══
        if self.model and ai_signal > 0.3:
            take_profit_pct = ai_signal * 1.5
        elif self.model and ai_signal < -0.3:
            take_profit_pct = ai_signal * 0.3
        else:
            take_profit_pct = 0.0

        # 7. Торговая логика
        if self.state["status"] != "running":
            return

        balance = self.state["balance"]
        bought = grid["bought_levels"]
        buys_this_cycle = 0  # Лимит покупок за цикл
        MAX_BUYS_PER_CYCLE = 2

        for lvl in grid["levels"]:
            lvl_key = str(lvl)

            # ═══ BUY ═══
            if price <= lvl and lvl_key not in bought and balance >= adjusted_order_size:

                # --- Лимит покупок за цикл ---
                if buys_this_cycle >= MAX_BUYS_PER_CYCLE:
                    break

                # --- COOLDOWN после стоп-лосса ---
                if self._last_stop_loss_time:
                    hours_since_sl = (datetime.now(timezone.utc) - self._last_stop_loss_time).total_seconds() / 3600
                    if hours_since_sl < self._stop_loss_cooldown_hours:
                        self._log_ai_decision("block_buy", f"кулдаун после стоп-лосса ({hours_since_sl:.1f}ч/{self._stop_loss_cooldown_hours}ч)", price, ai_signal)
                        continue

                # --- TREND FILTER: блокируем ВСЕ покупки в даунтренде ---
                if trend == "down":
                    self._log_ai_decision("block_buy", "даунтренд — покупки заблокированы", price, ai_signal)
                    continue

                # --- AI фильтр: блокируем при сильном bearish ---
                if ai_signal < -0.4:
                    self._log_ai_decision("block_buy", f"медвежий сигнал AI {ai_signal:.2f}", price, ai_signal)
                    continue

                # --- RSI фильтр: не покупаем при перекупленности ---
                if rsi > 75:
                    self._log_ai_decision("block_buy", f"RSI перекуплен {rsi:.0f}", price, ai_signal)
                    continue

                amount = adjusted_order_size / price
                allowed, reason = self.safety.check_can_buy(self.state, price, amount)
                if not allowed:
                    logger.info("🛡️ Покупка заблокирована: %s", reason)
                    continue

                order = self._place_order("buy", price, amount, lvl)
                if order:
                    bought[lvl_key] = {
                        "amount": amount,
                        "buy_price": price,
                        "buy_time": datetime.now(timezone.utc).isoformat(),
                        "order_id": order.get("id", ""),
                        "ai_signal_at_buy": round(ai_signal, 3),
                        "size_multiplier": round(size_multiplier, 2),
                        "peak_price": price,  # для trailing stop
                    }
                    self.state["balance"] -= adjusted_order_size
                    self._log_trade("buy", price, amount, order)
                    self._log_ai_decision("buy", f"signal={ai_signal:.2f}, rsi={rsi:.0f}, trend={trend}", price, ai_signal)
                    buys_this_cycle += 1

            # ═══ SELL ═══
            elif price >= lvl and lvl_key in bought:
                pos = bought[lvl_key]
                buy_price = pos["buy_price"]
                price_gain_pct = (price - buy_price) / buy_price * 100

                # ═══ AI: Динамический take-profit ═══
                if self.model and take_profit_pct > 0:
                    if price_gain_pct < take_profit_pct:
                        buy_time = pos.get("buy_time", "")
                        if buy_time:
                            try:
                                dt = datetime.fromisoformat(buy_time)
                                hold_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                                if hold_hours < 48:
                                    self._log_ai_decision("hold", f"gain {price_gain_pct:.1f}% < tp {take_profit_pct:.1f}%", price, ai_signal)
                                    continue
                            except Exception:
                                pass

                # --- Минимальная прибыль для продажи (чтобы не терять на комиссиях) ---
                min_profit_pct = self.config.fee_rate * 200 + 0.1  # комиссия x2 + 0.1% запас
                if 0 < price_gain_pct < min_profit_pct:
                    self._log_ai_decision("hold", f"прибыль {price_gain_pct:.2f}% < мин {min_profit_pct:.1f}%", price, ai_signal)
                    continue

                # --- RSI фильтр: при перепроданности не продаём (цена может отскочить) ---
                if rsi < 25 and price_gain_pct < 0:
                    self._log_ai_decision("hold_oversold", f"RSI={rsi:.0f}, ждём отскок", price, ai_signal)
                    continue

                # Стоп-лосс / trailing stop проверка
                force_sell = self.safety.check_should_force_sell(pos, price)
                if force_sell:
                    logger.warning("🔴 СТОП-ЛОСС: принудительная продажа!")
                    self._log_ai_decision("stop_loss", f"убыток {price_gain_pct:.1f}%", price, ai_signal)

                amount = pos["amount"]
                order = self._place_order("sell", price, amount, lvl)
                if order:
                    sell_value = amount * price
                    buy_cost = amount * buy_price
                    fee = sell_value * self.config.fee_rate
                    profit = sell_value - buy_cost - fee * 2
                    self.state["balance"] += sell_value
                    del bought[lvl_key]
                    self._log_trade("sell", price, amount, order, profit)
                    self._log_ai_decision("sell", f"прибыль={price_gain_pct:.1f}%, P&L=${profit:.2f}", price, ai_signal)

        # ═══ Trailing stop + стоп-лосс для всех позиций ═══
        for lvl_key in list(bought.keys()):
            pos = bought[lvl_key]

            # Обновляем peak_price для trailing stop
            if price > pos.get("peak_price", pos["buy_price"]):
                pos["peak_price"] = price

            # Проверяем trailing stop и обычный stop-loss
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
                    loss_pct = (pos["buy_price"] - price) / pos["buy_price"] * 100
                    self._log_ai_decision("forced_sell", f"стоп-лосс {loss_pct:.1f}%", price, ai_signal)
                    # Устанавливаем cooldown после стоп-лосса
                    self._last_stop_loss_time = datetime.now(timezone.utc)

        # 8. Обновляем equity
        holdings_value = sum(pos["amount"] * price for pos in bought.values())
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
            if len(curve) > 720:
                curve.pop(0)

    # ═══════════════════════════════════════════
    # Ордера
    # ═══════════════════════════════════════════

    def _round_amount(self, amount: float) -> float:
        """Округляет количество по правилам биржи."""
        precision = self.market_info.get("amount_precision", 8)
        if isinstance(precision, int):
            return float(self.exchange.decimal_to_precision(
                amount, 1, precision, 2))
        return round(amount, 8)

    def _round_price(self, price: float) -> float:
        """Округляет цену по правилам биржи."""
        precision = self.market_info.get("price_precision", 2)
        if isinstance(precision, int):
            return float(self.exchange.decimal_to_precision(
                price, 1, precision, 2))
        return round(price, 2)

    def _place_order(self, side: str, price: float, amount: float, level: float) -> dict:
        """Размещает ордер. В paper mode — симулирует."""
        try:
            amount = self._round_amount(amount)
            price = self._round_price(price)

            min_amount = self.market_info.get("min_amount", 0)
            min_cost = self.market_info.get("min_cost", 0)

            if min_amount and amount < min_amount:
                logger.warning("⚠️ Количество %.8f < мин %.8f для %s",
                             amount, min_amount, self.config.symbol)
                return None

            if min_cost and (amount * price) < min_cost:
                logger.warning("⚠️ Стоимость $%.2f < мин $%.2f для %s",
                             amount * price, min_cost, self.config.symbol)
                return None

            if amount <= 0:
                return None

            # ═══ Paper mode: симуляция ═══
            if self.paper_mode:
                order = {
                    "id": f"paper_{int(time.time()*1000)}",
                    "status": "closed",
                    "side": side,
                    "price": price,
                    "amount": amount,
                }
                logger.info("📝 PAPER %s: %.8f %s @ %.2f",
                           side.upper(), amount, self.config.symbol, price)
                return order

            # ═══ Реальный ордер ═══
            if side == "buy":
                order = self.exchange.create_limit_buy_order(
                    self.config.symbol, amount, price
                )
            else:
                order = self.exchange.create_limit_sell_order(
                    self.config.symbol, amount, price
                )

            logger.info("📋 %s ордер: %.8f %s @ %.2f (ID: %s)",
                       side.upper(), amount, self.config.symbol, price,
                       order.get("id", "?"))
            return order

        except ccxt.InsufficientFunds:
            logger.warning("💰 Недостаточно средств для %s %.8f @ %.2f", side, amount, price)
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
        if self.paper_mode:
            return
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
        if len(errors) > 100:
            errors.pop(0)
        self.state["error_log"] = errors

    # ═══════════════════════════════════════════
    # Запуск
    # ═══════════════════════════════════════════

    def reconcile(self):
        """Сверка состояния с биржей после перезапуска."""
        if self.paper_mode:
            logger.info("📝 Paper mode — сверка не требуется")
            return
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

        self.load_state()
        self.state["status"] = "running"

        self.reconcile()

        mode_str = "PAPER" if self.paper_mode else ("TESTNET" if self.config.testnet else "LIVE")
        logger.info("=" * 50)
        logger.info("🚀 СТАРТ: %s | Депозит: $%.2f | Сетка: %d уровней, %.1f%%",
                    self.config.symbol, self.config.investment,
                    self.config.grid_count, self.config.range_pct)
        logger.info("   Режим: %s | AI: %s | Интервал: %dс",
                    mode_str, bool(self.model),
                    self.config.poll_interval_seconds)
        logger.info("=" * 50)

        # Обработка сигналов завершения (только в main thread)
        import threading
        if threading.current_thread() is threading.main_thread():
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
                self.check_commands()

                if not self.running:
                    break

                action = self.safety.pre_cycle_check(self.state)
                if action == "pause":
                    self.state["status"] = "paused"
                    logger.warning("⏸️ Торговля приостановлена (безопасность)")
                elif action == "stop":
                    break

                if self.state["status"] == "running":
                    self.run_cycle()

                self.save_state()
                retry_delay = 1
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

    if not config.paper_trading and not config.api_key:
        print("❌ Установите переменные окружения:")
        print("   export BINANCE_API_KEY='ваш_ключ'")
        print("   export BINANCE_API_SECRET='ваш_секрет'")
        sys.exit(1)

    trader = LiveGridTrader(config)
    trader.start()


if __name__ == "__main__":
    main()
