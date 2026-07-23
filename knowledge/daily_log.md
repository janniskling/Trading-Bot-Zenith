# Daily Log – 2026-07-23


## Pre-Market Research
*12:20 UTC*

**Marktstimmung:** Neutral (SPY +0.1%, QQQ +0.1%)
**Futures:** SPY: +0.1% | QQQ: +0.1%

**News-Zusammenfassung:**
**GOOGL:**
  - [yahoo_finance] Google's extreme AI capex spending plans trigger a technical warning on the stock price
  - [yahoo_finance] Dow Jones Futures Fall As Oil Prices Top $90; Google, Tesla Skid On Earnings, Capital Spending
  - [yahoo_finance] Google Stock Falls Amid Questions Over AI Leadership, Gemini 4 Roadmap
**U:**
  - [yahoo_finance] Tesla stock slides after profit miss; full-year capex spend of $25 billion confirmed
  - [yahoo_finance] W.R. Berkley’s Q2 is Making Analysts Raise Price Targets, But Is The Optimism Justified?
  - [yahoo_finance] Santander UK halts branch cuts until 2028

**Top-Kandidaten:**
| Symbol | Score | Aktion | Grund |
|--------|-------|--------|-------|
| AAPL | 100/100 | watch_for_entry | EMA bullish crossover, Above EMA50, Volume 0.7x avg, RSI 62.9 |
| NVDA | 100/100 | watch_for_entry | EMA bullish crossover, Above EMA50, Volume 1.3x avg, RSI 56.2 |
| SPY | 100/100 | watch_for_entry | EMA bullish crossover, Above EMA50, Volume 0.8x avg, RSI 51.9 |
| META | 80/100 | watch_for_entry | EMA bullish crossover, Above EMA50, RSI 51.6 |
| AMD | 80/100 | watch_for_entry | EMA9 > EMA21 (no fresh cross), Above EMA50, Volume 1.0x avg, RSI 56.0 |

## Market Open – Orders
*15:33 UTC*

Keine Orders platziert.

## Midday Check
*17:54 UTC*

Positionen geprüft: 11
Aktionen: 0

## Nightly Reflection
*21:55 UTC*

**1 neue Lernerkenntnisse**

**Neue Erkenntnisse:**
- Reflection error: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash
Please retry in 23.229489025s. [links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count"
  quota_id: "GenerateContentInputTokensPerModelPerMinute-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
}
violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
}
violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests"
  quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
}
, retry_delay {
  seconds: 23
}
]
