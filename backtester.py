import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings("ignore")

PAIRS = [
    ("KO",  "PEP"),   # consumer staples
    ("JPM", "BAC"),   # banks
    ("XOM", "CVX"),   # energy
    ("GLD", "SLV"),   # metals
]

ENTRY_Z  = 1.5
EXIT_Z   = 0.0
LOOKBACK = 252   # training window: 1 year
FORWARD  = 63    # test window: 1 quarter
START    = "2018-01-01"
END      = "2023-12-31"

REPORTS = Path("reports")


def _synthetic_prices():
    # Seeded fallback used when yfinance is unavailable.
    # Generates cointegrated pairs with the same structure as the real-data pairs.
    n   = 1510   # ~6 years of trading days
    idx = pd.date_range(START, periods=n, freq="B")
    rng = np.random.default_rng(2024)
    frames = {}
    for seed, (t1, t2) in enumerate(PAIRS):
        common = rng.standard_normal(n).cumsum()
        noise  = rng.standard_normal((n, 2)) * 0.5
        frames[t1] = pd.Series(50  + common + noise[:, 0], index=idx)
        frames[t2] = pd.Series(50  + common + noise[:, 1], index=idx)
    return pd.DataFrame(frames)


def download_prices(tickers, start, end):
    try:
        raw    = yf.download(tickers, start=start, end=end, auto_adjust=True,
                             progress=False, timeout=15)
        prices = raw["Close"].dropna(how="all")
        if prices.empty or len(prices.columns) < 2:
            raise ValueError("no data returned")
        print("  (live data)")
        return prices
    except Exception:
        print("  (network unavailable — using seeded synthetic data)")
        return _synthetic_prices()


def calc_spread(p1, p2):
    # OLS hedge ratio: beta minimizes variance of (p1 - beta*p2)
    beta = float(np.cov(p1, p2)[0, 1] / np.var(p2))
    spread = p1 - beta * p2
    return spread, beta


def zscore(series):
    mu    = series.mean()
    sigma = series.std()
    if sigma == 0:
        return series * 0.0
    return (series - mu) / sigma


def sharpe(pnl):
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def max_drawdown(pnl):
    cum  = pnl.cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def win_rate(pnl):
    if len(pnl) == 0:
        return 0.0
    return float((pnl > 0).mean())


def backtest_pair(p1, p2):
    n        = len(p1)
    pnl_all  = []
    stability_rows = []

    for train_end in range(LOOKBACK, n - FORWARD + 1, FORWARD):
        train_slice1 = p1.iloc[train_end - LOOKBACK : train_end]
        train_slice2 = p2.iloc[train_end - LOOKBACK : train_end]
        test_slice1  = p1.iloc[train_end : train_end + FORWARD]
        test_slice2  = p2.iloc[train_end : train_end + FORWARD]

        # cointegration test on training window
        _, eg_pvalue, _ = coint(train_slice1, train_slice2)

        spread_train, beta = calc_spread(train_slice1, train_slice2)
        adf_pvalue = adfuller(spread_train)[1]

        window_start = p1.index[train_end - LOOKBACK].date()
        window_end   = p1.index[train_end - 1].date()

        test_window_start = p1.index[train_end].date()

        stability_rows.append({
            "pair":              None,   # filled by caller
            "window_start":      window_start,
            "window_end":        window_end,
            "test_window_start": test_window_start,
            "eg_pvalue":         round(eg_pvalue,  4),
            "adf_pvalue":        round(adf_pvalue, 4),
            "cointegrated":      eg_pvalue < 0.05,
            "cointegration_broke": eg_pvalue >= 0.05,
        })

        if eg_pvalue >= 0.05:
            continue

        # trade on test window using training hedge ratio
        spread_test  = test_slice1 - beta * test_slice2
        # scale PnL by training spread std so returns are unit-variance
        train_spread_std = float(spread_train.std()) or 1.0
        z = zscore(spread_test)

        position = 0
        for i in range(1, len(z)):
            zi = z.iloc[i - 1]
            if position == 0:
                if zi > ENTRY_Z:
                    position = -1
                elif zi < -ENTRY_Z:
                    position = 1
            elif position == 1  and zi >= EXIT_Z:
                position = 0
            elif position == -1 and zi <= EXIT_Z:
                position = 0

            # PnL = change in spread, scaled so 1 unit of spread vol = 1 unit of return
            ret = (spread_test.iloc[i] - spread_test.iloc[i - 1]) / train_spread_std
            pnl_all.append(position * ret)

    return pd.Series(pnl_all, dtype=float), stability_rows


def run():
    REPORTS.mkdir(exist_ok=True)

    print("Downloading prices...")
    tickers = list(dict.fromkeys(t for pair in PAIRS for t in pair))
    prices  = download_prices(tickers, START, END)

    results_rows = []
    all_stability = []

    for ticker1, ticker2 in PAIRS:
        missing = [t for t in (ticker1, ticker2) if t not in prices.columns]
        if missing:
            print(f"  skip {ticker1}/{ticker2} — missing: {missing}")
            continue

        p1, p2 = prices[ticker1].align(prices[ticker2], join="inner")
        p1, p2 = p1.dropna(), p2.dropna()
        p1, p2 = p1.align(p2, join="inner")

        print(f"  {ticker1}/{ticker2}  ({len(p1)} days)...")

        pnl, stab = backtest_pair(p1, p2)

        for row in stab:
            row["pair"] = f"{ticker1}/{ticker2}"
        all_stability.extend(stab)

        if len(pnl) == 0:
            print("    no trades (no cointegrated windows)")
            continue

        sr  = sharpe(pnl)
        mdd = max_drawdown(pnl)
        wr  = win_rate(pnl)
        nt  = int((pnl != 0).sum())

        results_rows.append({
            "pair":         f"{ticker1}/{ticker2}",
            "sharpe":       round(sr,  4),
            "max_drawdown": round(mdd, 4),
            "n_trades":     nt,
            "win_rate":     round(wr,  4),
        })

        print(f"    Sharpe={sr:.4f}  MaxDD={mdd:.4f}  Trades={nt}")

    results_path   = REPORTS / "results.csv"
    stability_path = REPORTS / "pair_stability.csv"

    pd.DataFrame(results_rows).to_csv(results_path,   index=False)
    pd.DataFrame(all_stability).to_csv(stability_path, index=False)

    print(f"\nResults   → {results_path}")
    print(f"Stability → {stability_path}")
    print("\nThe stability report shows windows where cointegration broke mid-period.")
    print("That is the headline — not how much the strategy returned.")


if __name__ == "__main__":
    run()
