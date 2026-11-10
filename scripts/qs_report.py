#!/usr/bin/env python3
"""QuantStats report runner.

Generic by design: ticker, series start and dividend events are inputs
(CLI flags); nothing about a specific market is baked into the workflow.

Data is fetched from Yahoo's chart API DIRECTLY with requests — never via
yfinance, whose Africa/Cairo timezone conversion crashes on the
2023-04-28 DST timestamp (NonExistentTimeError) and ignore_tz does not fix it.

Analyses:
  (1) Daily-returns analytics (QuantStats) over the window starting at
      --series-start: cumulative return, Sharpe, Sortino, max drawdown,
      daily win rate, volatility -> results/stats.json + QuantStats HTML
      tearsheet at results/tearsheet.html.
  (2) Dividend-capture event study recomputed from the fetched data:
      per event, net capture % (dividend credited) at ex-day close and at
      the close 2 sessions after ex-day (payday proxy); aggregates
      n / mean / win rate / min / max per exit -> results/stats.json and
      results/report.md.

Usage:
  python scripts/qs_report.py [--ticker POUL.CA] [--series-start 2026-09-28]
                              [--events events.json]
Events JSON format: [{"end": "YYYY-MM-DD", "ex": "YYYY-MM-DD", "div": 0.25}, ...]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

# Verified Cairo Poultry (POUL) distributions — the pilot inputs only;
# any market's events can be supplied via --events.
DEFAULT_EVENTS = [
    {"end": "2019-07-14", "ex": "2019-07-15", "div": 0.25},
    {"end": "2021-05-17", "ex": "2021-05-18", "div": 0.16},
    {"end": "2024-10-21", "ex": "2024-10-22", "div": 0.847875},
    {"end": "2026-04-26", "ex": "2026-04-27", "div": 0.33},
]

RESULTS = "results"


def fetch_chart(ticker, start="2018-01-01"):
    """Fetch daily OHLC from Yahoo's chart API directly; return (df, tz_name).

    Index is tz-aware in the exchange timezone (via tz_convert from UTC,
    which never hits nonexistent local midnights the way yfinance's
    localize path does). Index dates are exchange-local calendar dates.
    """
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history"
    )
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) qs-runner"},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()["chart"]["result"][0]
    tz_name = payload["meta"].get("exchangeTimezoneName", "UTC")
    ts = payload["timestamp"]
    q = payload["indicators"]["quote"][0]
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz_name)
    df = pd.DataFrame(
        {
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Volume": q.get("volume"),
        },
        index=idx,
    )
    df = df[df["Close"].notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df, tz_name


def qs_metrics(returns, prices):
    """QuantStats metrics with manual fallbacks so a single qs API quirk
    never kills the run."""
    import quantstats as qs

    out = {}
    out["sessions"] = int(len(prices))
    out["returns_days"] = int(len(returns))
    out["start"] = str(prices.index[0].date())
    out["end"] = str(prices.index[-1].date())
    out["start_close"] = float(prices.iloc[0])
    out["end_close"] = float(prices.iloc[-1])
    out["cumulative_return_pct"] = float((prices.iloc[-1] / prices.iloc[0] - 1.0) * 100.0)
    try:
        out["sharpe"] = float(qs.stats.sharpe(returns))
    except Exception:
        out["sharpe"] = float(
            returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        ) if returns.std(ddof=1) else None
    try:
        out["sortino"] = float(qs.stats.sortino(returns))
    except Exception:
        downside = returns[returns < 0]
        dd = downside.std(ddof=1) if len(downside) else np.nan
        out["sortino"] = float(returns.mean() / dd * np.sqrt(252)) if dd else None
    try:
        out["max_drawdown_pct"] = float(qs.stats.max_drawdown(returns) * 100.0)
    except Exception:
        curve = (1.0 + returns).cumprod()
        out["max_drawdown_pct"] = float((curve / curve.cummax() - 1.0).min() * 100.0)
    try:
        out["volatility_pct"] = float(qs.stats.volatility(returns) * 100.0)
    except Exception:
        out["volatility_pct"] = float(returns.std(ddof=1) * np.sqrt(252) * 100.0)
    out["daily_win_rate_pct"] = float((returns > 0).mean() * 100.0)
    return out


def capture_study(df, events):
    """Recompute per-event net capture % from the fetched sessions."""
    dates = [d.date() for d in df.index]
    rows = []
    for ev in events:
        end_d = pd.Timestamp(ev["end"]).date()
        ex_d = pd.Timestamp(ev["ex"]).date()
        div = float(ev["div"])
        cum_idx = max(i for i, d in enumerate(dates) if d <= end_d)
        ex_idx = min(i for i, d in enumerate(dates) if d >= ex_d)
        entry_close = float(df["Close"].iloc[cum_idx])
        ex_close = float(df["Close"].iloc[ex_idx])
        pay_idx = min(ex_idx + 2, len(df) - 1)
        pay_close = float(df["Close"].iloc[pay_idx])
        rows.append(
            {
                "end": ev["end"],
                "ex": ev["ex"],
                "div": div,
                "entry_session": str(dates[cum_idx]),
                "entry_close": entry_close,
                "ex_session": str(dates[ex_idx]),
                "ex_close": ex_close,
                "capture_ex_close_pct": (ex_close + div - entry_close) / entry_close * 100.0,
                "payday_session": str(dates[pay_idx]),
                "payday_close": pay_close,
                "capture_payday_pct": (pay_close + div - entry_close) / entry_close * 100.0,
            }
        )

    def agg(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {
            "n": int(len(vals)),
            "mean_pct": float(vals.mean()),
            "win_rate_pct": float((vals > 0).mean() * 100.0),
            "min_pct": float(vals.min()),
            "max_pct": float(vals.max()),
        }

    return {
        "events": rows,
        "ex_close": agg("capture_ex_close_pct"),
        "payday_close": agg("capture_payday_pct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="POUL.CA")
    ap.add_argument("--series-start", default="2026-09-28",
                    help="First session of the returns-analytics window (on/after).")
    ap.add_argument("--events", default=None, help="Optional events JSON file.")
    ap.add_argument("--fetch-start", default="2018-01-01")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    events = DEFAULT_EVENTS
    if args.events:
        with open(args.events) as f:
            events = json.load(f)

    print(f"Fetching {args.ticker} chart data directly from Yahoo ...")
    df, tz_name = fetch_chart(args.ticker, start=args.fetch_start)
    print(f"Rows: {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}  tz={tz_name}")
    df.to_csv(os.path.join(RESULTS, "ohlc.csv"))

    stats = {
        "ticker": args.ticker,
        "source": "Yahoo chart API (direct, via requests)",
        "exchange_tz": tz_name,
        "rows": int(len(df)),
        "range": [str(df.index[0].date()), str(df.index[-1].date())],
    }

    # ---- (1) returns analytics over the series window ----
    start_ts = pd.Timestamp(args.series_start).date()
    win = df[[d.date() >= start_ts for d in df.index]]
    prices = win["Close"].astype(float)
    returns = prices.pct_change().dropna()
    returns.name = args.ticker
    series = qs_metrics(returns, prices)
    stats["series"] = series
    print("Series metrics:", json.dumps(series, indent=2))

    tearsheet_status = "ok"
    try:
        import quantstats as qs

        qs.reports.html(
            returns,
            title=f"{args.ticker} — QuantStats tearsheet (from {series['start']})",
            output=os.path.join(RESULTS, "tearsheet.html"),
        )
    except Exception as e:  # keep the run alive; record what happened
        tearsheet_status = f"failed: {type(e).__name__}: {e}"
        print("Tearsheet generation failed:", tearsheet_status)
    stats["series"]["tearsheet"] = tearsheet_status

    # ---- (2) dividend-capture event study ----
    cap = capture_study(df, events)
    stats["capture"] = cap
    print("Capture aggregates:", json.dumps(
        {"ex_close": cap["ex_close"], "payday_close": cap["payday_close"]}, indent=2))

    with open(os.path.join(RESULTS, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- report.md ----
    s = series
    lines = [
        f"# QuantStats report — {args.ticker}",
        "",
        f"Data: Yahoo chart API (direct), {stats['rows']} sessions, "
        f"{stats['range'][0]} → {stats['range'][1]} ({tz_name}).",
        "",
        f"## (1) Daily returns from {s['start']} (window start {args.series_start})",
        "",
        f"- Sessions: {s['sessions']} ({s['returns_days']} daily returns)",
        f"- Close: {s['start_close']:.4f} → {s['end_close']:.4f}",
        f"- Cumulative return: {s['cumulative_return_pct']:+.2f}%",
        f"- Sharpe: {s['sharpe']:.4f}" if s["sharpe"] is not None else "- Sharpe: n/a",
        f"- Sortino: {s['sortino']:.4f}" if s["sortino"] is not None else "- Sortino: n/a",
        f"- Max drawdown: {s['max_drawdown_pct']:.2f}%",
        f"- Volatility (annualised): {s['volatility_pct']:.2f}%",
        f"- Daily win rate: {s['daily_win_rate_pct']:.1f}%",
        f"- Tearsheet: {tearsheet_status} (results/tearsheet.html)",
        "",
        "> ⚠ Short-window caveat: this window is only a few dozen sessions "
        "(post-merger trading resumption), so Sharpe/Sortino/volatility are "
        "annualised from very little data — treat them as indicative, not stable.",
        "",
        "## (2) Dividend-capture events (recomputed from the fetched data)",
        "",
        "| End | Ex | Div | Entry close | Ex close | Capture @ ex close | Payday close | Capture @ payday |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in cap["events"]:
        lines.append(
            f"| {r['end']} | {r['ex']} | {r['div']} | {r['entry_close']:.4f} | "
            f"{r['ex_close']:.4f} | {r['capture_ex_close_pct']:+.2f}% | "
            f"{r['payday_close']:.4f} | {r['capture_payday_pct']:+.2f}% |"
        )
    for label, key in (("Ex-day close", "ex_close"), ("Payday close (ex+2)", "payday_close")):
        a = cap[key]
        lines += [
            "",
            f"**{label}** — n={a['n']}, mean {a['mean_pct']:+.2f}%, "
            f"win rate {a['win_rate_pct']:.0f}%, min {a['min_pct']:+.2f}%, max {a['max_pct']:+.2f}%",
        ]
    lines += [
        "",
        "Capture = exit close + dividend − entry (last-cum) close, as % of entry close. "
        "Note: Yahoo's raw series is not merger-adjusted; % captures are scale-consistent "
        "per event. Events whose pre-2019 segment is corrupted in Yahoo's data are excluded "
        "from this runner's default event set.",
    ]
    with open(os.path.join(RESULTS, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote results/stats.json, results/report.md, results/ohlc.csv")


if __name__ == "__main__":
    main()
