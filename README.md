# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/egikumbara/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/egikumbara/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Earners-over-players forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase, a week after launch. It forecasts each of the 55 trophies' earn rate at the case's target as five quantiles, scored by pinball loss on the logit scale.

The model works in counts, not rates: a trophy's rate is 100 * E / N, where E is how many tracked players earned it and N is how many players Exophase tracks. E and N are fitted separately as power laws in time since release (log count against log(1 + hours since 2026-11-19T00:00Z)) and extrapolated from the last read, then divided — so early story trophies rise (E outgrows N) while late and grind trophies fall (N outgrows E). Slopes come from least squares over the case's distinct reads (runs of an unchanged page collapsed to one), are shrunk toward stillness by n / (n + 2), and are clamped so no extrapolation runs away; the median stays within 0.05x–20x of the last rate. The logit spread combines the baseline's random-walk term (0.5 * sqrt(days)), the fit's residual scatter, and the refresh jumps the history actually shows. When the whole history is one unchanged read (the practice cases), the page is stale and the model expects the refresh dilution the house documented for launch night (638 -> 3,868 players tracked; the sum of the rates fell 147.78 -> 110.55): each rate at 0.30x its last value, with a wide spread.

Verified locally: on the practice cases against a synthetic answer built from the documented 04:30 refresh aggregates (non-100% rates at 0.22x their launch values, reproducing the documented rate sum 110.55), it scores a pinball loss of **0.24** vs the starter baseline's **1.40** (typical miss 0.07 vs 0.68 points, 90% coverage 0.98 vs 0.00), and it beats the baseline for any uniform refresh factor up to 0.6. On a synthetic final-like case built from the real Exophase page of 20 November (Chapter 1 at 7.99%, most late trophies at 0.01–0.02%), the medians behave as the counts say: Chapter 1 rises, the fishing challenge rises, late trophies sit at the 0.01% floor. Two runs on the same cases are byte-identical; a cold run of both practice cases takes about 0.17 s (limit 8 s per case). Deterministic, Python 3.13 standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` reads every `<id>.in` case in the input directory and writes `<id>.out` with 55 rows (one per trophy, five quantiles each).

---

<sub>Entered on [make.fun](https://make.fun/?ref=q1q35jmc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
