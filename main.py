"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (fitted only on the case's own `comparables`, no outside data):
  1. Keep the rolling-midnight launches: tier "comparable" in the `unlock` note,
     hour 1 present, at least 20 of the 24 release-day hours present, and
     peak/hour-1 <= 20. This drops the simultaneous-global-unlock launches
     (Silksong, Battlefield 6, ...), whose hour 1 is pre-unlock and whose
     hour-1-normalised ratios are 90x-2000x, and the early-access and weak tiers.
  2. For each hour h = 2..24, regress log(viewers_h / viewers_1) on
     log(viewers_1) across those comparables (OLS, centred). Launch size is the
     dominant shape driver: bigger hour 1 -> flatter day (corr -0.82 between
     log hour-1 and log peak/hour-1); the biggest launches peak at hour 1.
  3. Quantiles come from the per-hour empirical quantiles of the regression
     residuals at 0.05/0.25/0.5/0.75/0.95, added to the fitted median.
  4. q_tau(h) = p1 * exp(mu_h + resid_tau_h), p1 = hour 1's print.

Local evaluation on the four practice cases (answers fetched from
SullyGnome's GameViewers API, hour-1 prints verified exact):
  practice loss 0.180 (baseline 0.387): p00 0.248, p01 0.168, p02 0.117, p03 0.186.
  Typical miss ~19% vs baseline ~83%; 90% coverage 0.91 vs 0.78.
On the final case the same code reads its 42 comparables (the 38 plus the
four practice launches, which are rolling-midnight console launches too)
and the same filter admits them.

Deterministic: stdlib only, no network, no randomness, no clock. Reads every
<input>/*.in, writes <id>.out per case: {"hours":[{"hour":h,"q":[...]}...]}.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
MAX_PEAK_RATIO = 20.0
MIN_HOURS = 20
MIN_COMPARABLES = 5
# Fallback (house baseline) spread, only if too few comparables pass the filter.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.15


def parse_tier(unlock):
    if not unlock:
        return "?"
    for tier in ("early-access-contaminated", "target-history", "comparable", "weak"):
        if "Tier: " + tier in unlock:
            return tier
    return "?"


def day_profile(comp):
    """{hour: viewers} for the 24 UTC hours of the comparable's release day."""
    day_start = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    out = {}
    for pt in comp.get("hourly") or []:
        viewers = pt.get("viewers")
        if viewers is None:
            continue
        t = datetime.fromisoformat(pt["from"].replace("Z", "+00:00"))
        dt = (t - day_start).total_seconds() / 3600
        if 0 <= dt < 24:
            out[int(dt) + 1] = viewers
    return out


def rolling_comparables(case):
    kept = []
    for comp in case.get("comparables") or []:
        if parse_tier(comp.get("unlock", "")) != "comparable":
            continue
        prof = day_profile(comp)
        h1 = prof.get(1)
        if not h1 or h1 <= 0 or len(prof) < MIN_HOURS:
            continue
        if max(prof.values()) / h1 > MAX_PEAK_RATIO:
            continue
        kept.append(prof)
    return kept


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


def round2(x):
    """Round to 2 decimals, half up (same as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    hours = [row["hour"] for row in case["forecast"]]
    comps = rolling_comparables(case)
    if len(comps) < MIN_COMPARABLES:
        # Fallback: house baseline persistence forecast.
        rows = []
        for h in hours:
            q = [round2(min(1e9, max(1, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))))) for z in Z]
            rows.append({"hour": h, "q": q})
        return {"hours": rows}

    log_p1 = math.log(p1)
    ref = sum(math.log(prof[1]) for prof in comps) / len(comps)
    rows = []
    for h in hours:
        xs, ys = [], []
        for prof in comps:
            if h in prof:
                xs.append(math.log(prof[1]) - ref)
                ys.append(math.log(prof[h] / prof[1]))
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        intercept = my - slope * mx
        resid = sorted(y - (intercept + slope * x) for x, y in zip(xs, ys))
        mu = intercept + slope * (log_p1 - ref)
        q = []
        for tau in LEVELS:
            v = p1 * math.exp(mu + quantile(resid, tau))
            q.append(min(1e9, max(1, v)))
        q.sort()
        rows.append({"hour": h, "q": [round2(v) for v in q]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
