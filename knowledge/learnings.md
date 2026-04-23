# Zenith Trading Learnings

<!-- Dieser File wird täglich vom Bot aktualisiert. Neue Erkenntnisse werden oben eingefügt. -->
<!-- Maximal 200 Einträge werden gespeichert. Ältere werden automatisch entfernt. -->

## Strategie-Regeln (fest, nicht überschreiben)
- Nur US-Aktien und ETFs — keine Optionsscheine
- Max. 5 Trades pro Tag (Käufe + Verkäufe zusammen)
- Max. 15% des Portfolios pro Position
- Ziel: ≤ 20% Cash idle; innerhalb von 3 Wochen voll investiert
- Stop-Loss: -3% unter Einstieg (Alpaca Bracket Order, automatisch)
- Take-Profit: +6% über Einstieg (2:1 Chance/Risiko, automatisch)
- Täglicher Verlust-Stopp: Kein neuer Trade wenn Portfolio -5% am Tag

## Erste Erkenntnisse (manuell eingetragen)
- EMA-Strategie braucht mindestens 60 Handelstage historische Daten — neue IPOs überspringen
- Vor Earnings (≤ 3 Tage) keine neuen Positionen eingehen — Whipsaw-Risiko zu hoch
- Volumenbestätigung ist kritisch: Nur bei Volume > 1.2x 20-Tage-Durchschnitt einsteigen
