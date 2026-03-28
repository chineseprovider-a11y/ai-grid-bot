"""Бэктест: Grid+AI гибридный бот на исторических данных."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from ai.feature_engineer import add_indicators, FEATURE_COLUMNS


@dataclass
class Trade:
    timestamp: str
    side: str        # "buy" or "sell"
    price: float
    amount: float
    fee: float
    profit: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    strategy: str   # "grid" or "ai_grid"
    total_profit: float
    total_fees: float
    net_profit: float
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    investment: float
    roi_pct: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


class GridBacktest:
    """Обычный Grid бот — бэктест."""

    def __init__(self, investment: float, grid_count: int, range_pct: float = 2.0,
                 fee_rate: float = 0.00075):
        self.investment = investment
        self.grid_count = grid_count
        self.range_pct = range_pct
        self.fee_rate = fee_rate

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        trades = []
        equity = []
        balance = self.investment
        holdings = {}  # price -> amount (bought positions)
        total_fees = 0.0
        total_profit = 0.0
        peak_balance = balance

        # Начальная настройка сетки
        start_price = df["close"].iloc[0]
        lower, upper, levels = self._setup_grid(start_price)
        order_size = self.investment / self.grid_count

        # Отслеживаем какие уровни были куплены
        bought_levels = set()

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            ts = str(df["timestamp"].iloc[i])

            # Цена вышла за диапазон — перенастройка
            if price < lower or price > upper:
                lower, upper, levels = self._setup_grid(price)
                bought_levels.clear()
                continue

            for lvl in levels:
                # BUY: цена опустилась до уровня, который ещё не куплен
                if price <= lvl and lvl not in bought_levels and balance >= order_size:
                    amount = order_size / price
                    fee = order_size * self.fee_rate
                    balance -= order_size
                    total_fees += fee
                    bought_levels.add(lvl)
                    holdings[lvl] = {"amount": amount, "buy_price": price}
                    trades.append(Trade(ts, "buy", price, amount, fee))

                # SELL: цена поднялась выше уровня, который был куплен
                elif price >= lvl and lvl in bought_levels and lvl in holdings:
                    pos = holdings.pop(lvl)
                    sell_value = pos["amount"] * price
                    fee = sell_value * self.fee_rate
                    buy_cost = pos["amount"] * pos["buy_price"]
                    profit = sell_value - buy_cost - fee - (buy_cost * self.fee_rate)
                    balance += sell_value
                    total_fees += fee
                    total_profit += profit
                    bought_levels.discard(lvl)
                    trades.append(Trade(ts, "sell", price, pos["amount"], fee, profit))

            # Equity = cash + holdings value
            holdings_value = sum(p["amount"] * price for p in holdings.values())
            total_equity = balance + holdings_value
            equity.append(total_equity)
            peak_balance = max(peak_balance, total_equity)

        net_profit = total_profit - total_fees
        win_trades = sum(1 for t in trades if t.side == "sell" and t.profit > 0)
        sell_trades = sum(1 for t in trades if t.side == "sell")
        win_rate = win_trades / sell_trades if sell_trades > 0 else 0

        # Max drawdown
        max_dd = self._max_drawdown(equity) if equity else 0

        # Sharpe ratio (hourly returns)
        sharpe = self._sharpe_ratio(equity) if len(equity) > 10 else 0

        return BacktestResult(
            symbol=symbol,
            strategy="grid",
            total_profit=round(total_profit, 4),
            total_fees=round(total_fees, 4),
            net_profit=round(net_profit, 4),
            total_trades=len(trades),
            win_trades=win_trades,
            loss_trades=sell_trades - win_trades,
            win_rate=round(win_rate, 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4),
            investment=self.investment,
            roi_pct=round(net_profit / self.investment * 100, 2),
            trades=trades,
            equity_curve=equity,
        )

    def _setup_grid(self, price):
        lower = price * (1 - self.range_pct / 100)
        upper = price * (1 + self.range_pct / 100)
        step = (upper - lower) / self.grid_count
        levels = [round(lower + i * step, 8) for i in range(1, self.grid_count)]
        return lower, upper, levels

    def _max_drawdown(self, equity):
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    def _sharpe_ratio(self, equity, risk_free=0):
        returns = pd.Series(equity).pct_change().dropna()
        if returns.std() == 0:
            return 0
        return (returns.mean() - risk_free) / returns.std() * np.sqrt(24 * 365)


class AIGridBacktest(GridBacktest):
    """
    Grid + AI бот v2 — AI работает как умный фильтр поверх стандартного Grid.

    Принципы:
    1. Сетка НИКОГДА не перестраивается из-за AI (только при выходе цены за диапазон)
    2. AI только ФИЛЬТРУЕТ сделки: пропускает плохие покупки, задерживает продажи
    3. Размер позиций одинаковый — без множителей
    4. Используем скользящее среднее предсказаний для устойчивости
    """

    def __init__(self, investment: float, grid_count: int, range_pct: float = 2.0,
                 fee_rate: float = 0.00075, model=None, scaler=None, seq_length: int = 48):
        super().__init__(investment, grid_count, range_pct, fee_rate)
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        # Добавляем индикаторы для AI предсказаний
        df_ai = add_indicators(df.copy()).dropna().reset_index(drop=True)

        trades = []
        equity = []
        balance = self.investment
        holdings = {}
        total_fees = 0.0
        total_profit = 0.0

        # Стандартная сетка — как в обычном Grid
        start_price = df_ai["close"].iloc[self.seq_length]
        lower, upper, levels = self._setup_grid(start_price)
        order_size = self.investment / self.grid_count
        bought_levels = set()

        # AI состояние — скользящее среднее предсказаний
        prediction_history = []  # хранит последние предсказания
        last_prediction_i = 0
        ai_signal = 0.0  # -1.0 (bearish) .. 0.0 (neutral) .. +1.0 (bullish)

        # Статистика точности AI для самокалибровки
        ai_correct = 0
        ai_total = 0
        ai_accuracy = 0.5  # начинаем с нейтральной

        for i in range(self.seq_length, len(df_ai)):
            price = df_ai["close"].iloc[i]
            ts = str(df_ai["timestamp"].iloc[i])

            # AI предсказание каждые 12 часов (реже = стабильнее)
            if self.model and (i - last_prediction_i) >= 12:
                prediction = self._predict(df_ai, i)
                if prediction:
                    # Проверяем предыдущее предсказание (самокалибровка)
                    if prediction_history and last_prediction_i > 0:
                        prev_pred = prediction_history[-1]
                        prev_price = df_ai["close"].iloc[last_prediction_i]
                        actual_return = (price - prev_price) / prev_price

                        was_correct = False
                        if prev_pred > 0.2 and actual_return > 0:
                            was_correct = True
                        elif prev_pred < -0.2 and actual_return < 0:
                            was_correct = True
                        elif abs(prev_pred) <= 0.2 and abs(actual_return) < 0.005:
                            was_correct = True

                        ai_total += 1
                        if was_correct:
                            ai_correct += 1
                        ai_accuracy = ai_correct / ai_total if ai_total > 0 else 0.5

                    # Конвертируем вероятности в непрерывный сигнал [-1, +1]
                    probs = prediction["probabilities"]
                    raw_signal = probs["bullish"] - probs["bearish"]

                    # Скользящее среднее сигнала (сглаживание шума)
                    prediction_history.append(raw_signal)
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)

                    # Взвешенное среднее: последние предсказания важнее
                    weights = list(range(1, len(prediction_history) + 1))
                    ai_signal = sum(s * w for s, w in zip(prediction_history, weights)) / sum(weights)

                    # Снижаем влияние AI если он часто ошибается
                    if ai_accuracy < 0.4:
                        ai_signal *= 0.3  # AI плохо работает — почти игнорируем
                    elif ai_accuracy < 0.5:
                        ai_signal *= 0.6  # AI неуверенный — частично игнорируем

                    last_prediction_i = i

            # Цена вышла за диапазон — стандартная перенастройка
            if price < lower or price > upper:
                lower, upper, levels = self._setup_grid(price)
                bought_levels.clear()
                continue

            # Торговля с AI-фильтром
            for lvl in levels:
                # ═══ BUY FILTER ═══
                if price <= lvl and lvl not in bought_levels and balance >= order_size:
                    # AI фильтр: пропускаем покупку только при СИЛЬНОМ bearish сигнале
                    if ai_signal < -0.4:
                        continue  # сильный bearish — не покупаем

                    # RSI фильтр: не покупаем при перекупленности
                    rsi = df_ai["rsi"].iloc[i] if "rsi" in df_ai.columns else 50
                    if rsi > 75 and ai_signal < 0:
                        continue  # перекуплено + bearish — пропускаем

                    amount = order_size / price
                    fee = order_size * self.fee_rate
                    balance -= order_size
                    total_fees += fee
                    bought_levels.add(lvl)
                    holdings[lvl] = {"amount": amount, "buy_price": price, "buy_i": i}
                    trades.append(Trade(ts, "buy", price, amount, fee))

                # ═══ SELL FILTER ═══
                elif price >= lvl and lvl in bought_levels and lvl in holdings:
                    pos = holdings[lvl]
                    hold_duration = i - pos.get("buy_i", i)

                    # AI фильтр продажи: задерживаем продажу при СИЛЬНОМ bullish
                    if ai_signal > 0.5 and hold_duration < 24:
                        # Сильный bullish сигнал + держим < 24ч → подождём
                        # Но продаём если цена выросла достаточно (> 1% от покупки)
                        price_gain = (price - pos["buy_price"]) / pos["buy_price"]
                        if price_gain < 0.01:
                            continue  # подождём ещё

                    # Стандартная продажа
                    pos = holdings.pop(lvl)
                    sell_value = pos["amount"] * price
                    fee = sell_value * self.fee_rate
                    buy_cost = pos["amount"] * pos["buy_price"]
                    profit = sell_value - buy_cost - fee - (buy_cost * self.fee_rate)
                    balance += sell_value
                    total_fees += fee
                    total_profit += profit
                    bought_levels.discard(lvl)
                    trades.append(Trade(ts, "sell", price, pos["amount"], fee, profit))

            holdings_value = sum(p["amount"] * price for p in holdings.values())
            equity.append(balance + holdings_value)

        net_profit = total_profit - total_fees
        win_trades = sum(1 for t in trades if t.side == "sell" and t.profit > 0)
        sell_trades = sum(1 for t in trades if t.side == "sell")
        win_rate = win_trades / sell_trades if sell_trades > 0 else 0
        max_dd = self._max_drawdown(equity) if equity else 0
        sharpe = self._sharpe_ratio(equity) if len(equity) > 10 else 0

        return BacktestResult(
            symbol=symbol,
            strategy="ai_grid",
            total_profit=round(total_profit, 4),
            total_fees=round(total_fees, 4),
            net_profit=round(net_profit, 4),
            total_trades=len(trades),
            win_trades=win_trades,
            loss_trades=sell_trades - win_trades,
            win_rate=round(win_rate, 4),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4),
            investment=self.investment,
            roi_pct=round(net_profit / self.investment * 100, 2),
            trades=trades,
            equity_curve=equity,
        )

    def _predict(self, df, i):
        """Предсказание AI на основе последних seq_length свечей."""
        try:
            features = df[FEATURE_COLUMNS].iloc[i - self.seq_length:i].values
            if len(features) < self.seq_length:
                return None
            scaled = self.scaler.transform(features)
            X = np.expand_dims(scaled, axis=0)
            probs = self.model.predict(X, verbose=0)[0]
            return {
                "probabilities": {
                    "bearish": float(probs[0]),
                    "sideways": float(probs[1]),
                    "bullish": float(probs[2]),
                },
            }
        except Exception:
            return None


@dataclass
class OptimizationResult:
    """Результат оптимизации одной комбинации параметров."""
    grid_count: int
    range_pct: float
    net_profit: float
    roi_pct: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float


class GridOptimizer:
    """
    Перебор параметров Grid бота для поиска оптимальной конфигурации.
    Оптимизирует grid_count и range_pct по выбранной метрике.
    """

    def __init__(self, investment: float,
                 grid_counts: list = None,
                 range_pcts: list = None,
                 fee_rate: float = 0.00075,
                 use_ai: bool = False,
                 model=None, scaler=None):
        self.investment = investment
        self.grid_counts = grid_counts or list(range(5, 26, 5))  # [5, 10, 15, 20, 25]
        self.range_pcts = range_pcts or [round(x * 0.5, 1) for x in range(2, 11)]  # [1.0..5.0]
        self.fee_rate = fee_rate
        self.use_ai = use_ai
        self.model = model
        self.scaler = scaler

    def optimize(self, df: pd.DataFrame, symbol: str,
                 metric: str = "roi_pct",
                 progress_callback=None) -> list:
        """
        Прогоняет бэктест для всех комбинаций параметров.
        metric: "roi_pct", "sharpe_ratio", "net_profit", "win_rate"
        progress_callback: callable(current, total) для отображения прогресса.
        Возвращает список OptimizationResult, отсортированный по metric (desc).
        """
        combos = list(product(self.grid_counts, self.range_pcts))
        results = []

        for idx, (gc, rp) in enumerate(combos):
            if self.use_ai and self.model:
                bt = AIGridBacktest(self.investment, gc, rp, self.fee_rate,
                                    model=self.model, scaler=self.scaler)
            else:
                bt = GridBacktest(self.investment, gc, rp, self.fee_rate)

            try:
                res = bt.run(df, symbol)
                results.append(OptimizationResult(
                    grid_count=gc,
                    range_pct=rp,
                    net_profit=res.net_profit,
                    roi_pct=res.roi_pct,
                    total_trades=res.total_trades,
                    win_rate=res.win_rate,
                    max_drawdown=res.max_drawdown,
                    sharpe_ratio=res.sharpe_ratio,
                ))
            except Exception:
                pass

            if progress_callback:
                progress_callback(idx + 1, len(combos))

        results.sort(key=lambda r: getattr(r, metric), reverse=True)
        return results

    def results_to_dataframe(self, results: list) -> pd.DataFrame:
        """Конвертирует результаты в DataFrame для отображения."""
        return pd.DataFrame([
            {
                "Grid уровней": r.grid_count,
                "Диапазон %": r.range_pct,
                "Прибыль $": round(r.net_profit, 2),
                "ROI %": round(r.roi_pct, 2),
                "Сделок": r.total_trades,
                "Win Rate %": round(r.win_rate * 100, 1),
                "Max DD %": round(r.max_drawdown * 100, 2),
                "Sharpe": round(r.sharpe_ratio, 2),
            }
            for r in results
        ])

    def results_to_heatmap(self, results: list, metric: str = "roi_pct") -> pd.DataFrame:
        """Создаёт pivot-таблицу для heatmap визуализации."""
        data = []
        for r in results:
            data.append({
                "grid_count": r.grid_count,
                "range_pct": r.range_pct,
                "value": getattr(r, metric),
            })
        df = pd.DataFrame(data)
        return df.pivot(index="grid_count", columns="range_pct", values="value")
