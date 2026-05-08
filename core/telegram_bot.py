from __future__ import annotations

from datetime import date

import requests

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from core.alpaca_client import Position, Order


MAX_MESSAGE_LEN = 4000

_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class TelegramNotifier:
    def __init__(self) -> None:
        self._chat_id = TELEGRAM_CHAT_ID

    def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        chunks = [text] if len(text) <= MAX_MESSAGE_LEN else _split_message(text, MAX_MESSAGE_LEN)
        for chunk in chunks:
            try:
                resp = requests.post(
                    f"{_API_BASE}/sendMessage",
                    json={"chat_id": self._chat_id, "text": chunk, "parse_mode": parse_mode},
                    timeout=10,
                )
                resp.raise_for_status()
            except Exception as e:
                print(f"Telegram send_message failed: {e}")

    def send_premarket_brief(self, candidates: list[dict], market_mood: str) -> None:
        today = date.today().strftime("%d.%m.%Y")
        lines = [
            f"*Zenith Pre-Market Brief – {today}*",
            f"📊 Marktstimmung: {market_mood}",
            "",
            "*Top-Kandidaten heute:*",
        ]
        if candidates:
            for c in candidates[:5]:
                score_bar = "🟢" if c.get("score", 0) >= 60 else "🟡"
                lines.append(f"{score_bar} *{c['symbol']}* ({c.get('score', 0)}/100) – {c.get('reason', '')}")
        else:
            lines.append("_Keine starken Signale heute_")
        self.send_message("\n".join(lines))

    def send_market_open_summary(self, orders: list[Order]) -> None:
        today = date.today().strftime("%d.%m.%Y")
        lines = [f"*Zenith – Marktöffnung {today}*", ""]
        if orders:
            lines.append("*Platzierte Orders:*")
            for o in orders:
                if o.status == "dry_run":
                    lines.append(f"  🔵 [DRY RUN] {o.symbol} – {o.qty:.0f} Stk")
                else:
                    lines.append(f"  📥 BUY {o.symbol} – {o.qty:.0f} Stk @ Limit")
        else:
            lines.append("_Keine Orders platziert (kein Signal oder Tageslimit erreicht)_")
        self.send_message("\n".join(lines))

    def send_midday_check(self, positions: list[Position]) -> None:
        today = date.today().strftime("%d.%m.%Y")
        lines = [f"*Zenith Midday Check – {today}*", ""]
        if positions:
            lines.append("*Offene Positionen:*")
            for p in positions:
                emoji = "📈" if p.unrealized_pl >= 0 else "📉"
                pct = p.unrealized_plpc * 100
                lines.append(
                    f"  {emoji} *{p.symbol}*: `{p.unrealized_pl:+.2f}€` ({pct:+.1f}%) "
                    f"| {p.qty:.0f} Stk @ {p.avg_entry_price:.2f}"
                )
        else:
            lines.append("_Keine offenen Positionen_")
        self.send_message("\n".join(lines))

    def send_midday_actions(self, actions: list[str]) -> None:
        if not actions:
            return
        lines = ["*Zenith – Midday Aktionen:*"] + [f"  • {a}" for a in actions]
        self.send_message("\n".join(lines))

    def send_daily_summary(
        self,
        portfolio: dict,
        closed_trades: list[dict],
        positions: list[Position],
        trade_count: int,
    ) -> None:
        today = date.today().strftime("%d.%m.%Y")
        equity = portfolio.get("equity", 0)
        cash_pct = portfolio.get("cash", 0) / equity * 100 if equity else 0
        day_pnl = portfolio.get("day_pnl", 0)
        day_pnl_pct = portfolio.get("day_pnl_pct", 0)
        pnl_emoji = "📈" if day_pnl >= 0 else "📉"

        lines = [
            f"*Zenith Daily Report – {today}*",
            f"{pnl_emoji} Portfolio: `{equity:,.2f}€` ({day_pnl_pct:+.2f}%)",
            f"💵 Cash-Anteil: {cash_pct:.1f}%",
            "",
        ]

        if closed_trades:
            lines.append("*Geschlossene Trades heute:*")
            wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
            for t in closed_trades:
                outcome = "✅" if t.get("pnl", 0) > 0 else "❌"
                lines.append(
                    f"  {outcome} {t['symbol']}: `{t.get('pnl', 0):+.2f}€` "
                    f"({t.get('pnl_pct', 0):+.1f}%)"
                )
            win_rate = wins / len(closed_trades) * 100 if closed_trades else 0
            lines.append(f"\n_Win-Rate: {win_rate:.0f}% ({wins}/{len(closed_trades)})_")
        else:
            lines.append("_Keine Trades heute geschlossen_")

        if positions:
            lines.append("\n*Offene Positionen:*")
            for p in positions:
                pct = p.unrealized_plpc * 100
                emoji = "📈" if p.unrealized_pl >= 0 else "📉"
                lines.append(f"  {emoji} {p.symbol}: `{p.unrealized_pl:+.2f}€` ({pct:+.1f}%)")

        lines.append(f"\n_Trades heute: {trade_count}/{5}_")
        self.send_message("\n".join(lines))

    def send_reflection_complete(self, new_lessons: int) -> None:
        msg = (
            f"🧠 *Zenith Nightly Reflection abgeschlossen*\n"
            f"{new_lessons} neue Lernerkenntnisse zu `learnings.md` hinzugefügt."
        )
        self.send_message(msg)

    def send_error(self, step: str, error: str) -> None:
        msg = f"⚠️ *Zenith Fehler im Schritt `{step}`*\n```{error[:500]}```"
        self.send_message(msg)


def _split_message(text: str, max_len: int) -> list[str]:
    chunks: list[str] = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks
