# quantstats-runner

Runs **QuantStats** portfolio/returns analytics via GitHub Actions — the
commands to execute live in the `COMMANDS` repository variable and the
workflow is triggered manually (`workflow_dispatch` only).

## Pattern

1. Edit the `COMMANDS` repository variable (Settings → Secrets and
   variables → Actions → Variables), e.g.:

   ```
   mkdir -p results
   python scripts/qs_report.py 2>&1 | tee results/run.log
   ```

2. Run the workflow manually from the Actions tab (or via the API).
3. Download the `results` artifact from the run.

## The pilot analysis (scripts/qs_report.py)

Generic by design — ticker, series window and dividend events are inputs
(`--ticker`, `--series-start`, `--events`), so the same repo serves any
market. Defaults are the Cairo Poultry (POUL.CA) pilot:

1. **Returns analytics** — daily returns from 2026-09-28 (post-merger
   trading resumption): cumulative return, Sharpe, Sortino, max drawdown,
   daily win rate, volatility → `results/stats.json`, plus a QuantStats
   HTML tearsheet at `results/tearsheet.html`.
2. **Dividend-capture event study** — per-event net capture % at ex-day
   close and at the close 2 sessions after ex-day, with n / mean / win
   rate / min / max per exit → `results/stats.json`, `results/report.md`.

Price data is fetched from Yahoo's chart API **directly** with `requests`
— never via yfinance, whose Africa/Cairo timezone conversion crashes on
the 2023-04-28 DST timestamp (`NonExistentTimeError`; `ignore_tz` does
not fix it). Yahoo's raw series is not merger-adjusted.
