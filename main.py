"""Earners-over-players trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at
the case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) in percent.

The forecast is built entirely from the case file, in three ideas:

  1. Earners over players, not rates. A trophy's earn rate is a ratio,
     rate = 100 * E / N, where E is how many tracked players earned the
     trophy and N is how many players Exophase tracks. The two move for
     different reasons: E grows as players play through the game, N grows
     as new players buy it and sync a profile. A rate that falls does not
     mean players un-earn a trophy; it means N outgrew E. So the model
     fits the two counts separately, each as a power law in time since
     release (log count against log(1 + hours since 2026-11-19T00:00Z)),
     and extrapolates both to the target from the last read: E from its
     own fitted slope, N from its own, then divides. Early story trophies
     (E outgrows N) rise; late and grind trophies (N outgrows E) fall.
     That is exactly what the public page shows between launch night and
     20 November: Chapter 1 went 1.25% -> 7.99% while most late trophies
     fell to 0.01-0.02%, the same handful of early earners diluted by a
     far larger tracked base.

  2. Anchored extrapolation, slopes shrunk toward stillness. The slopes
     come from least squares over the case's distinct reads (runs of an
     unchanged page collapsed to one read, so a page Exophase left
     unrefreshed for hours does not count as many reads), but the levels
     are anchored at the last read: a convex history must not drag the
     forecast through its stale opening values. Each earner slope is
     shrunk toward 0 (persistence) by n / (n + 2), so a trophy with two
     reads barely moves and one with two hundred is trusted. Slopes are
     clamped so no extrapolation runs away, and the median is kept within
     0.05x-20x of the last rate.

  3. Honest spread. The logit spread combines the baseline's random-walk
     term (0.5 * sqrt(days)), the fit's residual scatter, and the refresh
     jumps the history actually shows (Exophase refreshes a game's counts
     on its own schedule, hours apart, so consecutive distinct reads jump).
     The pinball loss rewards quantiles you really believe, so the band
     widens where the history disagrees with itself and stays tight where
     it does not.

  Stale page. When the whole history is one unchanged read (the practice
  cases: every read identical, 638 players tracked), there is no trend to
  fit and the page is stale: Exophase has not refreshed since the first
  read. The next refresh replaces the tracked cohort with a much larger
  one, and the documented first refresh did exactly that (638 -> 3,868
  players; the sum of the rates fell 147.78 -> 110.55, and excluding the
  trophy that stayed at 100%, 47.78 -> 10.55, a factor of 0.22). So a
  flat history forecasts each rate at 0.30x its last value (kept at 100%
  where it already is 100%), with a wide spread. This branch never fires
  on the final, whose history spans two days of real reads.

Deterministic: no randomness, no clock, no network; the same case always
yields the same bytes. Python 3.13 standard library only.

Usage (outside the grader's container):
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (there is none; this file is whole).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# GTA VI's release: the anchor of the log-time axis (2026-11-19T00:00:00Z).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Calibration constants (see the module docstring for where each comes from) ---
# Smoothing added to an earner count before taking its log (a trophy with no
# earner yet still has a defined, small log-count).
SMOOTH = 0.5
# Pseudo-count for slope shrinkage toward 0: a slope from n reads is scaled
# by n / (n + SHRINK_PRIOR_N).
SHRINK_PRIOR_N = 2.0
# Random-walk spread, in logit units per square-root day (the house baseline's).
RW_SIGMA = 0.5
# Fit-residual spread is floored and capped to this logit range. The cap is
# tight on purpose: a fit's residual mostly measures refresh noise the jump
# term already counts, not trend uncertainty.
FIT_FLOOR = 0.20
FIT_CAP = 0.6
# Refresh-jump spread used when the history shows fewer than two jumps:
# the launch-night refresh moved most rates by roughly this much in logit.
JUMP_PRIOR = 0.6
# A refresh jump is a step toward the truth, not a random walk: only half
# its size counts toward the spread, capped.
JUMP_WEIGHT = 0.5
JUMP_CAP = 0.6
# Overall spread is kept inside this logit range.
SPREAD_MIN = 0.25
SPREAD_MAX = 1.5
# Stale-page branch: factor applied to the last rate, and its spread.
STALE_FACTOR = 0.30
STALE_SIGMA = 0.9
# Safety rails on the extrapolated median, relative to the trophy's last rate.
RATE_FLOOR_FACTOR = 0.05
RATE_CEIL_FACTOR = 20.0
# Clamps on the fitted log-time slopes (earners, players).
SLOPE_E_MIN = -1.0
SLOPE_E_MAX = 4.0
SLOPE_N_MIN = 0.0
SLOPE_N_MAX = 2.5


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct, 0.01, 99.99)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(when: datetime) -> float:
    """log(1 + hours since release); pre-release reads sit at 0."""
    hours = (when - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y ~ a + b*x. Returns (a, b, residual std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    # Distinct reads: collapse runs of identical (players, rates) so a page
    # Exophase left unrefreshed for hours does not count as many reads.
    distinct: list[dict] = []
    for read in history:
        if distinct:
            prev = distinct[-1]
            if (read.get("players_tracked") == prev.get("players_tracked")
                    and read.get("rates") == prev.get("rates")):
                continue
        distinct.append(read)

    flat = len(distinct) < 2
    last_read = distinct[-1]
    x_last = log_time(parse_time(last_read["at"]))
    dx = x_target - x_last
    n_last = last_read.get("players_tracked")
    n_last = float(n_last) if isinstance(n_last, (int, float)) and n_last > 0 else 638.0

    # Shared players-tracked slope: log N against log-time over distinct reads,
    # anchored at the last read's count.
    n_xs: list[float] = []
    n_ys: list[float] = []
    for read in distinct:
        players = read.get("players_tracked")
        if isinstance(players, (int, float)) and players > 0:
            n_xs.append(log_time(parse_time(read["at"])))
            n_ys.append(math.log(float(players)))
    _n_a, n_b, _n_resid = ols(n_xs, n_ys)
    n_b = clamp(n_b, SLOPE_N_MIN, SLOPE_N_MAX)
    players_pred = n_last * math.exp(n_b * dx) if dx > 0 else n_last

    # Refresh jumps actually seen: logit moves between consecutive distinct
    # reads, per trophy, pooled. Few or none seen -> the JUMP_PRIOR below.
    jumps: list[float] = []
    for i in range(len(trophies)):
        prev_lg: float | None = None
        for read in distinct:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if not isinstance(rate, (int, float)):
                continue
            lg = logit(float(rate))
            if prev_lg is not None and lg != prev_lg:
                jumps.append(abs(lg - prev_lg))
            prev_lg = lg
    if len(jumps) >= 2:
        jump_sigma = clamp(median(jumps) * JUMP_WEIGHT, 0.1, JUMP_CAP)
    else:
        jump_sigma = JUMP_PRIOR

    rw_sigma = RW_SIGMA * math.sqrt(horizon_days)

    rows = []
    for i, trophy in enumerate(trophies):
        xs: list[float] = []
        ys: list[float] = []
        last_rate: float | None = None
        for read in distinct:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if not isinstance(rate, (int, float)):
                continue
            players = read.get("players_tracked")
            if not isinstance(players, (int, float)) or players <= 0:
                continue
            last_rate = float(rate)
            earners = float(rate) * float(players) / 100.0
            xs.append(log_time(parse_time(read["at"])))
            ys.append(math.log(earners + SMOOTH))
        if last_rate is None:
            last_rate = 0.0
        _a, b_raw, resid = ols(xs, ys)
        n_pts = len(xs)

        if flat:
            # Stale page: no trend exists, so expect the refresh dilution.
            med = 100.0 if last_rate >= 99.99 else max(last_rate * STALE_FACTOR, 0.01)
            sigma = STALE_SIGMA
        elif n_pts < 2 or dx <= 0:
            # Too little to extrapolate: persistence with a baseline spread.
            med = clamp(last_rate, 0.01, 99.99)
            sigma = clamp(math.sqrt(rw_sigma ** 2 + 0.5 ** 2 + jump_sigma ** 2),
                          SPREAD_MIN, SPREAD_MAX)
        else:
            # Earner slope, shrunk toward 0, anchored at the last read.
            b = clamp(b_raw, SLOPE_E_MIN, SLOPE_E_MAX)
            b *= n_pts / (n_pts + SHRINK_PRIOR_N)
            earners_last = max(last_rate * n_last / 100.0, SMOOTH)
            earners_pred = earners_last * math.exp(b * dx)
            med = 100.0 * earners_pred / players_pred if players_pred > 0 else last_rate
            med = clamp(med, 0.01, 99.99)
            if last_rate >= 99.99:
                med = 99.99
            elif last_rate > 0:
                med = clamp(med, last_rate * RATE_FLOOR_FACTOR,
                            last_rate * RATE_CEIL_FACTOR)
                med = clamp(med, 0.01, 99.99)
            # Spread: random walk + fit scatter + refresh jumps.
            fit_sigma = clamp(resid, FIT_FLOOR, FIT_CAP) if n_pts >= 3 else 0.5
            sigma = math.sqrt(rw_sigma ** 2 + fit_sigma ** 2 + jump_sigma ** 2)
            sigma = clamp(sigma, SPREAD_MIN, SPREAD_MAX)

        lg = logit(med)
        qs: list[float | int] = []
        prev = 0.0
        for z in Z:
            q = clamp(inv_logit(lg + z * sigma), 0.01, 99.99)
            q = round4(max(q, prev))
            qs.append(q)
            prev = float(q)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_rows(case: dict) -> dict:
    """Persistence fallback: last rate, baseline spread. Never invalid."""
    rows = []
    history = case.get("history") or []
    last = history[-1] if history else {}
    rates = last.get("rates") or []
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    spread = RW_SIGMA * math.sqrt(horizon_days) if horizon_days > 0 else 0.5
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i] if i < len(rates) and isinstance(rates[i], (int, float)) else 0.5
        lg = logit(float(rate))
        qs = [round4(clamp(inv_logit(lg + z * spread), 0.01, 99.99)) for z in Z]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_rows(case)
            except Exception:
                forecast = {"trophies": []}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
