from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class EntrySignal:
    symbol: str
    entry_price: float
    confidence_score: int  # 0-100
    reason: str
    atr_stop: float = 0.0  # ATR-based stop price; 0 = fall back to fixed %


@dataclass
class ExitSignal:
    symbol: str
    reason: str


@dataclass
class ScoredCandidate:
    symbol: str
    score: int  # 0-100
    signals_met: list[str]


class BaseStrategy(ABC):
    @abstractmethod
    def check_entry_signal(self, symbol: str) -> EntrySignal | None:
        ...

    @abstractmethod
    def check_exit_signal(self, symbol: str, bars: pd.DataFrame) -> ExitSignal | None:
        ...

    @abstractmethod
    def score_candidates(self, symbols: list[str]) -> list[ScoredCandidate]:
        ...
