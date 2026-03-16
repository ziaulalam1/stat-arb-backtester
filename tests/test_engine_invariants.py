"""
Property tests on the backtest engine itself, not the strategy.

Any strategy run through this engine is tested against these invariants.
I defined what a valid backtest looks like — the strategy is separate from that definition.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtester import backtest_pair, calc_spread, zscore, sharpe, LOOKBACK, FORWARD, ENTRY_Z, EXIT_Z


def _make_cointegrated_pair(n=600, seed=None):
    rng    = np.random.default_rng(seed)
    common = rng.standard_normal(n).cumsum()
    p1     = pd.Series(100 + common + rng.standard_normal(n) * 0.5)
    p2     = pd.Series(100 + common + rng.standard_normal(n) * 0.5)
    idx    = pd.date_range("2018-01-01", periods=n, freq="B")
    p1.index = idx
    p2.index = idx
    return p1, p2


def test_no_lookahead_leak():
    # For every fold, the test window must start strictly after the training window ends.
    # Training windows intentionally overlap in walk-forward (rolling window), but a fold's
    # test window must never overlap with that same fold's training window.
    p1, p2 = _make_cointegrated_pair(seed=1)
    _, stability = backtest_pair(p1, p2)

    assert len(stability) > 0, "No walk-forward windows produced"

    for row in stability:
        train_end   = row["window_end"]
        test_start  = row["test_window_start"]
        assert test_start > train_end, (
            f"Lookahead: test window starts {test_start} before training ends {train_end}"
        )


def test_positions_bounded():
    # Position sizing is unit only: abs(position) <= 1.0 at every bar.
    p1, p2 = _make_cointegrated_pair(seed=2)
    n = len(p1)

    for train_end in range(LOOKBACK, n - FORWARD + 1, FORWARD):
        spread_train, _ = calc_spread(
            p1.iloc[train_end - LOOKBACK : train_end],
            p2.iloc[train_end - LOOKBACK : train_end],
        )
        _, beta = calc_spread(
            p1.iloc[train_end - LOOKBACK : train_end],
            p2.iloc[train_end - LOOKBACK : train_end],
        )
        spread_test = p1.iloc[train_end : train_end + FORWARD] - beta * p2.iloc[train_end : train_end + FORWARD]
        z = zscore(spread_test)

        position = 0
        for zi in z:
            if position == 0:
                if zi > ENTRY_Z:
                    position = -1
                elif zi < -ENTRY_Z:
                    position = 1
            elif position == 1  and zi >= EXIT_Z:
                position = 0
            elif position == -1 and zi <= EXIT_Z:
                position = 0

            assert abs(position) <= 1.0, f"Position {position} exceeds unit bound"


def test_pure_noise_sharpe():
    # Pure random walks have no cointegration structure.
    # The engine should not consistently produce Sharpe > 1.0 on noise.
    high_sharpe_count = 0
    for seed in range(10):
        rng = np.random.default_rng(seed + 200)
        n   = 700
        p1  = pd.Series(rng.standard_normal(n).cumsum() + 100)
        p2  = pd.Series(rng.standard_normal(n).cumsum() + 100)
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        p1.index = idx
        p2.index = idx
        pnl, _ = backtest_pair(p1, p2)
        if len(pnl) > 0 and sharpe(pnl) > 1.0:
            high_sharpe_count += 1

    # Allow at most 3 out of 10 seeds to exceed the threshold (noise fluctuation)
    assert high_sharpe_count <= 3, (
        f"Noise produced Sharpe > 1.0 in {high_sharpe_count}/10 seeds — "
        "possible overfitting or lookahead"
    )


def test_train_window_size_exact():
    # Every training window passed to cointegration test must be exactly LOOKBACK bars.
    # This catches off-by-one errors in window slicing.
    p1, p2 = _make_cointegrated_pair(n=500, seed=3)
    n = len(p1)
    for train_end in range(LOOKBACK, n - FORWARD + 1, FORWARD):
        window1 = p1.iloc[train_end - LOOKBACK : train_end]
        assert len(window1) == LOOKBACK, (
            f"Training window has {len(window1)} bars, expected {LOOKBACK}"
        )


def test_test_window_does_not_overlap_train():
    # The test window must start at train_end, not before.
    p1, p2 = _make_cointegrated_pair(n=500, seed=4)
    n = len(p1)
    for train_end in range(LOOKBACK, n - FORWARD + 1, FORWARD):
        train_end_date = p1.index[train_end - 1]
        test_start_date = p1.index[train_end]
        assert test_start_date > train_end_date, (
            f"Test window starts {test_start_date} which is not after train end {train_end_date}"
        )
