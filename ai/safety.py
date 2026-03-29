"""Система безопасности для live-торговли."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("SafetyGuard")


class SafetyGuard:
    """
    Проверяет каждый ордер перед отправкой на биржу.
    Единая точка контроля рисков.
    """

    def __init__(self, config):
        self.max_drawdown_pct = config.max_drawdown_pct
        self.stop_loss_pct = config.stop_loss_per_position_pct
        self.trailing_stop_pct = config.trailing_stop_pct
        self.max_positions = config.max_open_positions
        self.max_position_value_pct = config.max_position_value_pct
        self.daily_loss_limit = config.daily_loss_limit

    def check_can_buy(self, state: dict, price: float, amount: float) -> tuple:
        """
        Можно ли открыть новую позицию?
        Возвращает (allowed: bool, reason: str).
        """
        if state.get("status") != "running":
            return False, f"Бот не запущен (статус: {state.get('status')})"

        # Проверка просадки
        current_dd = state.get("safety", {}).get("current_drawdown", 0)
        if current_dd * 100 >= self.max_drawdown_pct:
            return False, f"Просадка {current_dd*100:.1f}% >= лимит {self.max_drawdown_pct}%"

        # Проверка кол-ва открытых позиций
        open_positions = len(state.get("grid", {}).get("bought_levels", {}))
        if open_positions >= self.max_positions:
            return False, f"Открыто {open_positions} позиций (макс {self.max_positions})"

        # Проверка объёма позиций
        investment = state.get("initial_investment", 0)
        position_value = state.get("safety", {}).get("total_position_value", 0)
        new_value = price * amount
        if investment > 0:
            pct = (position_value + new_value) / investment * 100
            if pct > self.max_position_value_pct:
                return False, f"Позиции {pct:.0f}% депозита (макс {self.max_position_value_pct}%)"

        # Дневной лимит убытков
        today_loss = self._get_today_loss(state)
        if today_loss >= self.daily_loss_limit:
            return False, f"Дневной убыток ${today_loss:.2f} >= лимит ${self.daily_loss_limit}"

        return True, "OK"

    def check_should_force_sell(self, position: dict, current_price: float) -> bool:
        """
        Нужно ли экстренно продать позицию?
        Проверяет:
        1. Обычный стоп-лосс (от цены покупки)
        2. Trailing stop (от пиковой цены)
        """
        buy_price = position.get("buy_price", 0)
        if buy_price <= 0:
            return False

        # 1. Обычный стоп-лосс от цены покупки
        loss_from_buy = (buy_price - current_price) / buy_price * 100
        if loss_from_buy >= self.stop_loss_pct:
            logger.warning(
                "STOP-LOSS: куплено по %.2f, сейчас %.2f (убыток %.1f%%)",
                buy_price, current_price, loss_from_buy,
            )
            return True

        # 2. Trailing stop: если цена была выше и упала
        peak_price = position.get("peak_price", buy_price)
        if peak_price > buy_price * 1.01:  # Активируем trailing только после +1% роста
            drop_from_peak = (peak_price - current_price) / peak_price * 100
            if drop_from_peak >= self.trailing_stop_pct:
                # Но только если мы всё ещё в прибыли или потеряли немного
                profit_from_buy = (current_price - buy_price) / buy_price * 100
                logger.warning(
                    "TRAILING STOP: пик %.2f, сейчас %.2f (откат %.1f%% от пика, прибыль %.1f%% от покупки)",
                    peak_price, current_price, drop_from_peak, profit_from_buy,
                )
                return True

        return False

    def pre_cycle_check(self, state: dict) -> str:
        """
        Проверка перед каждым циклом.
        Возвращает: 'ok', 'pause', 'stop'.
        """
        if state.get("status") == "stopped":
            return "stop"

        peak = state.get("safety", {}).get("peak_equity", 0)
        current = state.get("current_equity", 0)

        if peak > 0 and current > 0:
            dd = (peak - current) / peak
            if dd * 100 >= self.max_drawdown_pct:
                logger.critical(
                    "КРИТИЧЕСКАЯ ПРОСАДКА: %.1f%% — торговля остановлена!",
                    dd * 100,
                )
                return "pause"

        return "ok"

    def _get_today_loss(self, state: dict) -> float:
        """Подсчёт убытков за сегодня."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total_loss = 0.0
        for trade in state.get("trade_history", []):
            if trade.get("side") == "sell" and trade.get("profit", 0) < 0:
                ts = trade.get("timestamp", "")
                if ts.startswith(today):
                    total_loss += abs(trade["profit"])
        return total_loss
