# Backtester: Interview Explain Sheet

## What It Does

This backtester reports two things: what the strategy returned, and when its assumptions
stopped holding. The stability report is the headline. Returns without assumption checks
are not useful.

It downloads equity prices via yfinance, screens pairs for cointegration using
Engle-Granger, computes a spread z-score, and runs a walk-forward backtest. For each
pair, it reports Sharpe ratio, max drawdown, trade count, and win rate. It also logs
every walk-forward window with the ADF p-value and a flag for windows where cointegration
broke mid-period.

---

## 15 Questions With Locked Answers

**1. Why Engle-Granger and not Johansen?**

Engle-Granger is for two assets. Johansen is for three or more. This backtester tests
pairs, so Engle-Granger is the right tool. Johansen on a pair is not wrong, but it adds
complexity without adding anything. Two-asset cointegration has one cointegrating vector
by definition.

**2. Why ±1.5 sigma as the entry threshold?**

It is the published consensus in the academic literature on pairs trading. ±1.0 sigma
generates too many trades that revert by a small amount and get wiped out by transaction
costs. ±2.0 misses most mean reversion events because the spread rarely reaches that far.
±1.5 is the sweet spot between signal quality and trade frequency.

**3. Why walk-forward and not a single train/test split?**

A single split tells you how the strategy performed on one period. Walk-forward simulates
deployment: retrain periodically on the most recent history, apply to the next unseen
window, advance, repeat. It catches regime changes and gives you a distribution of
out-of-sample results instead of one number.

**4. Why report both Sharpe and max drawdown?**

Sharpe is a volatility-normalized return. It treats upside and downside variance the
same. Max drawdown captures the worst peak-to-trough loss, which is what a risk manager
actually cares about. A strategy can have a good Sharpe and a catastrophic drawdown if
the losses are clustered. You need both.

**5. What does the spread represent and why does it matter?**

The spread is p1 minus beta times p2. Beta is the OLS hedge ratio — it is the slope of
p1 regressed on p2. A cointegrated pair has a spread that is stationary: it drifts away
from zero but gets pulled back. The z-score of the spread tells you how far it has moved
from its historical mean in units of standard deviations.

**6. How is the hedge ratio computed?**

OLS: beta = Cov(p1, p2) / Var(p2). This minimizes the residual variance of p1 - beta*p2
on the training window. The hedge ratio is locked in at training time and applied
unchanged to the test window. Recomputing it during the test window would be lookahead.

**7. When does a trade enter and exit?**

Enter long the spread when z < -1.5. Enter short when z > +1.5. Exit either side when z
crosses zero. The logic is symmetric. There is no stop loss in this version — that is
listed in limitations.

**8. How is PnL calculated?**

Daily PnL is position times the change in spread, divided by the training-window spread
standard deviation. The divisor normalizes across pairs with different spread scales so
you can compare Sharpe ratios directly.

**9. What is the pair stability report and why is it the headline?**

For each walk-forward window, the stability report records the Engle-Granger p-value
and a flag for windows where cointegration was lost. A pair that looks profitable might
only have traded in its cointegrated windows and stopped generating signals when
cointegration broke. The stability report shows that directly. Returns without this are
misleading.

**10. What does the ADF p-value in the stability report mean?**

ADF is the Augmented Dickey-Fuller test. It tests whether the spread has a unit root.
Low p-value means the spread is stationary, which is the necessary condition for
mean-reversion to work. The ADF is run on the training spread as a secondary check
alongside Engle-Granger.

**11. What does the no-lookahead invariant in the engine tests prove?**

For every fold, the test window must start strictly after the training window ends. The
invariant test checks this directly: it reads test_window_start and window_end from each
stability row and asserts test_window_start > window_end. If the slicing had an
off-by-one error, this test would catch it.

**12. What does the position-bound invariant prove?**

Positions are always exactly 0, 1, or -1 — never larger. The invariant walks through the
same position logic the engine uses and asserts abs(position) <= 1.0 at every bar. This
rules out a class of bugs where the engine double-counts or fails to close a position.

**13. What does the noise sanity check prove?**

It runs the engine on pure random walks with no cointegration structure. The test asserts
that no more than 3 out of 10 seeds produce Sharpe > 1.0. If the engine were overfitting
or leaking future data, random noise would generate strong Sharpe ratios consistently.
The fact that it doesn't is a basic correctness signal.

**14. What would you change if this were going into production?**

Transaction costs first — bid-ask spread and market impact. Then dynamic hedge ratio
updates on a separate rolling window, not locked at training time. Then a stop loss to
avoid holding a losing spread indefinitely. Then factor neutralization so the spread is
not carrying latent market beta.

**15. Why is the engine tested separately from the strategy?**

The engine is a correctness harness. Any strategy that runs through it gets the same
invariant checks: no lookahead, bounded positions, noise sanity. The strategy logic is
separate. If you swap in a different signal, the invariants still apply. This is the
difference between testing "did this strategy work" and testing "is this a valid
backtest."
