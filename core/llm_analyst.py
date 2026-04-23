from __future__ import annotations

import json
from dataclasses import dataclass, field

import google.generativeai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL

_REFLECTION_SYSTEM = """You are Zenith's trading intelligence — a disciplined, data-driven paper trading bot using a Triple-Confirmation EMA strategy (EMA9/21/50 crossover + volume + RSI) on US stocks and ETFs.

During nightly reflection, extract concrete, quantitative lessons from today's log.

Rules:
- Only stock/ETF trades — no options or warrants
- Max 5 trades/day (buys + sells combined)
- Max 15% of portfolio per position
- Target: ≤20% cash idle; fully invested within 3 weeks

Response format (valid JSON only, no markdown wrapping):
{
  "new_learnings": ["lesson with numbers where possible", ...],
  "watchlist_updates": {"SYMBOL": "updated thesis note", ...},
  "tomorrow_focus": "one sentence",
  "risk_flags": ["concern if any"]
}"""

_ANALYSIS_SYSTEM = """You are Zenith's pre-market analyst. You rank stocks from a watchlist for today's trading session based on technical scores, news sentiment, and learned patterns.

Rules: stocks/ETFs only, no options. Return a JSON array of trade candidates ranked by priority.

Response format (valid JSON array):
[
  {
    "symbol": "AAPL",
    "score": 75,
    "reason": "EMA bullish crossover, volume 1.4x avg, positive news",
    "risk": "Earnings in 2 days — reduce position size",
    "action": "watch_for_entry"
  }
]
Possible actions: "strong_entry", "watch_for_entry", "avoid_today", "add_to_watchlist"."""


@dataclass
class TradeCandidate:
    symbol: str
    score: int
    reason: str
    risk: str = ""
    action: str = "watch_for_entry"


@dataclass
class ReflectionResult:
    new_learnings: list[str] = field(default_factory=list)
    watchlist_updates: dict[str, str] = field(default_factory=dict)
    tomorrow_focus: str = ""
    risk_flags: list[str] = field(default_factory=list)


class LLMAnalyst:
    def __init__(self) -> None:
        genai.configure(api_key=GEMINI_API_KEY)

    def _call(self, system: str, user_message: str, max_tokens: int = 1024) -> str:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        response = model.generate_content(user_message)
        return response.text.strip()

    def analyze_trade_candidates(
        self,
        watchlist_content: str,
        learnings_content: str,
        news_summary: str,
        scored_candidates: list,
        is_underdeployed: bool = False,
    ) -> list[TradeCandidate]:
        underdeployed_note = (
            "\n⚡ CAPITAL DEPLOYMENT PRIORITY: Cash > 20% of portfolio. "
            "Relax signal thresholds — accept candidates with confidence ≥ 60/100."
            if is_underdeployed else ""
        )

        scored_text = "\n".join(
            f"  {c.symbol}: {c.score}/100 – {', '.join(c.signals_met)}"
            for c in scored_candidates
        )

        user_message = f"""Today's watchlist and technical scores:
{scored_text}

News summary:
{news_summary}
{underdeployed_note}

Rank the top candidates for today and return a JSON array.
Learnings context:
{learnings_content[:3000]}"""

        try:
            text = self._call(_ANALYSIS_SYSTEM, user_message, max_tokens=1024)
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            data = json.loads(text)
            return [
                TradeCandidate(
                    symbol=item["symbol"],
                    score=item.get("score", 50),
                    reason=item.get("reason", ""),
                    risk=item.get("risk", ""),
                    action=item.get("action", "watch_for_entry"),
                )
                for item in data
            ]
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return [
                TradeCandidate(symbol=c.symbol, score=c.score, reason=", ".join(c.signals_met))
                for c in scored_candidates[:5]
            ]

    def reflect_on_day(
        self,
        daily_log_content: str,
        learnings_content: str,
    ) -> ReflectionResult:
        user_message = f"""<daily_log>
{daily_log_content}
</daily_log>

<existing_learnings>
{learnings_content[:4000]}
</existing_learnings>

Reflect on today's trading. Extract NEW lessons not already in existing_learnings. Be specific and quantitative where possible."""

        try:
            text = self._call(_REFLECTION_SYSTEM, user_message, max_tokens=2048)
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            data = json.loads(text)
            return ReflectionResult(
                new_learnings=data.get("new_learnings", []),
                watchlist_updates=data.get("watchlist_updates", {}),
                tomorrow_focus=data.get("tomorrow_focus", ""),
                risk_flags=data.get("risk_flags", []),
            )
        except Exception as e:
            print(f"LLM reflection failed: {e}")
            return ReflectionResult(new_learnings=[f"Reflection error: {e}"])

    def format_candidates_for_log(self, candidates: list[TradeCandidate]) -> str:
        if not candidates:
            return "Keine Kandidaten identifiziert."
        lines = ["| Symbol | Score | Aktion | Grund |", "|--------|-------|--------|-------|"]
        for c in candidates:
            lines.append(f"| {c.symbol} | {c.score}/100 | {c.action} | {c.reason} |")
        return "\n".join(lines)
