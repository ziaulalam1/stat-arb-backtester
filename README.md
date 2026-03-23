# Statistical Arbitrage Signal Backtester

Statistical arbitrage relies on a premise that eventually breaks. The question is not whether the spread reverts — it is whether the cointegration that makes it revert is still holding. This backtester reports both: what the strategy returned, and per walk-forward window, whether the assumption was still valid when it traded.

In this run, three of four pairs — KO/PEP, JPM/BAC, GLD/SLV — never passed the cointegration gate in any window. The backtester produced no trades for those pairs. XOM/CVX was the only tradeable pair. A system that does not surface this produces returns averaged across cointegrated and non-cointegrated windows, which is not a useful number. The stability report makes the difference visible.

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
| `reports/xom_cvx_zscore.png` | XOM/CVX spread z-score across cointegrated walk-forward windows — entry/exit signals shaded |

## Structure

```
backtester.py                   entry point — download, test, backtest, report
tests/test_backtester.py        spread calc, z-score, signal logic, metrics
tests/test_engine_invariants.py no-lookahead, position bounds, noise sanity check
```

## Design decisions

- **Engle-Granger, not Johansen** — pairs are two assets; Johansen adds complexity without adding information. Two-asset cointegration has one cointegrating vector by definition.
- **±1.5σ entry** — at ±1.0, the spread crosses threshold on normal noise too frequently. At ±2.0, XOM/CVX produces roughly half the 49 trades seen at ±1.5σ with no Sharpe improvement. ±1.5 is the empirically stable point.
- **Walk-forward, not single split** — simulates deployment: retrain on the most recent year, apply to the next quarter, advance. Produces a distribution of out-of-sample results instead of one number and catches regime changes.
- **Sharpe + max drawdown** — Sharpe normalizes by volatility but treats upside and downside variance the same. Max drawdown captures clustered losses. A strategy can have a positive Sharpe with a catastrophic drawdown if losses are concentrated.
- **Stability tracking alongside returns** — cointegration check on every window, not just initial screening. Returns without per-window assumption validation are not interpretable.

## Limitations

No transaction costs. No stop loss. Hedge ratio is fixed at training time. Universe is
four pairs. See INTERVIEW.md for full tradeoffs.
