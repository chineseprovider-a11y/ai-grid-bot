"""Конфигурация live-торговли."""

import os
import json
from dataclasses import dataclass, asdict

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "live_config.json")


@dataclass
class SafetyConfig:
    max_drawdown_pct: float = 15.0       # Макс. просадка — стоп торговли
    stop_loss_per_position_pct: float = 8.0  # Стоп-лосс на одну позицию
    trailing_stop_pct: float = 3.0       # Trailing stop: откат от пика (%)
    max_open_positions: int = 8          # Макс. кол-во открытых позиций
    max_position_value_pct: float = 80.0  # Макс. % депозита в позициях
    daily_loss_limit: float = 250.0      # Макс. убыток за день ($)


@dataclass
class LiveConfig:
    symbol: str = "BTC/USDT"
    investment: float = 5000.0
    grid_count: int = 8
    range_pct: float = 5.0
    fee_rate: float = 0.001
    paper_trading: bool = True           # Paper trading: реальные цены, симуляция ордеров
    testnet: bool = False                # Binance Testnet (фейковые цены)
    poll_interval_seconds: int = 60      # Как часто проверять рынок
    ai_prediction_interval_hours: int = 1  # AI прогноз каждый час
    use_ai: bool = True
    safety: SafetyConfig = None

    def __post_init__(self):
        if self.safety is None:
            self.safety = SafetyConfig()
        elif isinstance(self.safety, dict):
            self.safety = SafetyConfig(**{k: v for k, v in self.safety.items()
                                          if k in SafetyConfig.__dataclass_fields__})

    @property
    def api_key(self) -> str:
        return os.environ.get("BINANCE_API_KEY", "")

    @property
    def api_secret(self) -> str:
        return os.environ.get("BINANCE_API_SECRET", "")

    def save(self, path: str = None):
        path = path or CONFIG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = asdict(self)
        data.pop("api_key", None)
        data.pop("api_secret", None)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str = None) -> "LiveConfig":
        path = path or CONFIG_PATH
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})
