# Statistical Arbitrage Signal Backtester

This backtester reports two things — what the strategy returned, and when its assumptions stopped holding.

Downloads equity prices via yfinance, screens pairs for cointegration using Engle-Granger,
computes a spread z-score, and runs a walk-forward backtest. For every walk-forward window
it also tracks ADF p-values and flags windows where cointegration broke mid-period.
That stability report is the headline. Returns without it are not useful.

## Demo

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python backtester.py
```

## Tests

```bash
pytest
```

## Outputs

| File | What it contains |
|------|-----------------|
| `reports/results.csv` | Per-pair: Sharpe, max drawdown, trade count, win rate |
| `reports/pair_stability.csv` | Per walk-forward window: EG p-value, ADF p-value, cointegration flag |

## Structure

```
backtester.py                   entry point — download, test, backtest, report
tests/test_backtester.py        spread calc, z-score, signal logic, metrics
tests/test_engine_invariants.py no-lookahead, position bounds, noise sanity check
```

## Design decisions

- **Engle-Granger, not Johansen** — pairs are two assets; EG is the right test
- **±1.5σ entry** — published consensus; ±1.0 overtrades, ±2.0 misses most reversions
- **Walk-forward, not single split** — simulates deployment; catches regime changes
- **Sharpe + max drawdown** — Sharpe misses clustered losses; drawdown captures them

## Limitations

No transaction costs. No stop loss. Hedge ratio is fixed at training time. Universe is
four pairs. See INTERVIEW.md for full tradeoffs.
