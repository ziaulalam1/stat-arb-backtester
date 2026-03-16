import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtester import calc_spread, zscore, sharpe, max_drawdown, win_rate, ENTRY_Z, EXIT_Z


def test_spread_calculation():
    # beta should recover close to 2.0
    rng = np.random.default_rng(42)
    p2 = pd.Series(100 + np.cumsum(rng.standard_normal(300)))
    p1 = pd.Series(2.0 * p2.values + rng.standard_normal(300) * 0.5)
    spread, beta = calc_spread(p1, p2)
    assert abs(beta - 2.0) < 0.3, f"Hedge ratio {beta:.3f} far from 2.0"
    assert len(spread) == len(p1)


def test_spread_length_matches_input():
    rng = np.random.default_rng(7)
    p1 = pd.Series(rng.standard_normal(100).cumsum() + 50)
    p2 = pd.Series(rng.standard_normal(100).cumsum() + 50)
    spread, _ = calc_spread(p1, p2)
    assert len(spread) == 100


def test_zscore_normalization():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(500) * 10 + 50)
    z = zscore(s)
    assert abs(z.mean()) < 0.01
    assert abs(z.std() - 1.0) < 0.01


def test_zscore_constant_series():
    z = zscore(pd.Series([5.0] * 100))
    assert (z == 0).all()


def test_signal_long_entry():
    # z < -ENTRY_Z triggers long; exits when z >= EXIT_Z
    z = pd.Series([-2.0, -2.0, -2.0, 0.5, 0.5])
    position  = 0
    positions = []
    for zi in z:
        if position == 0 and zi < -ENTRY_Z:
            position = 1
        elif position == 1 and zi >= EXIT_Z:
            position = 0
        positions.append(position)
    assert positions[0] == 1   # entered on first bar
    assert positions[3] == 0   # exited when z > 0


def test_signal_short_entry():
    # z > ENTRY_Z triggers short; exits when z <= EXIT_Z
    z = pd.Series([2.0, 2.0, 2.0, -0.5, -0.5])
    position  = 0
    positions = []
    for zi in z:
        if position == 0 and zi > ENTRY_Z:
            position = -1
        elif position == -1 and zi <= EXIT_Z:
            position = 0
        positions.append(position)
    assert positions[0] == -1  # entered short
    assert positions[3] ==  0  # exited when z < 0


def test_no_trade_inside_band():
    # z never crosses ±ENTRY_Z — no position should be taken
    z = pd.Series([0.5, -0.5, 0.3, -0.3])
    position = 0
    for zi in z:
        if position == 0 and zi > ENTRY_Z:
            position = -1
        elif position == 0 and zi < -ENTRY_Z:
            position = 1
    assert position == 0


def test_sharpe_zero_variance():
    assert sharpe(pd.Series([0.0] * 100)) == 0.0


def test_sharpe_positive_pnl():
    rng = np.random.default_rng(9)
    # positive mean, nonzero variance
    pnl = pd.Series(rng.standard_normal(252) * 0.005 + 0.002)
    s   = sharpe(pnl)
    assert s > 0


def test_max_drawdown_never_positive():
    rng = np.random.default_rng(3)
    pnl = pd.Series(rng.standard_normal(300))
    assert max_drawdown(pnl) <= 0.0


def test_max_drawdown_monotone_up():
    # monotonically rising equity — drawdown is zero
    pnl = pd.Series([0.01] * 100)
    assert max_drawdown(pnl) == pytest.approx(0.0, abs=1e-9)


def test_win_rate_bounds():
    rng = np.random.default_rng(5)
    pnl = pd.Series(rng.standard_normal(200))
    wr  = win_rate(pnl)
    assert 0.0 <= wr <= 1.0


def test_win_rate_all_wins():
    assert win_rate(pd.Series([1.0] * 50)) == pytest.approx(1.0)


def test_win_rate_all_losses():
    assert win_rate(pd.Series([-1.0] * 50)) == pytest.approx(0.0)
