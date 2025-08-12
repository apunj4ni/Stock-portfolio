#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:39:19 2025

@author: aryanpunjani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJIA max-Sharpe portfolio build, clustering, and benchmark comparison.
Tested with Python 3.11, yfinance>=0.2, pandas>=2.0, numpy, scipy, matplotlib, seaborn (optional).
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# seaborn is optional; comment out if you prefer pure matplotlib
try:
    import seaborn as sns
    sns.set_context("talk")
    sns.set_style("whitegrid")
except Exception:
    pass

from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# =========================
# Config
# =========================
import pandas as pd
START = "2015-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
RISK_FREE_ANNUAL = 0.01             # 1% annual RF
TRADING_DAYS = 252
RISK_FREE_DAILY = RISK_FREE_ANNUAL / TRADING_DAYS
TOP_N = 10
LONG_ONLY_BOUNDS = (0.0, 1.0)       # long-only portfolio

DJIA_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]
BENCH1 = "SPY"
BENCH2 = "DIA"   # or '^DJI' for index

# =========================
# Helpers
# =========================
def download_prices(tickers, start=START, end=END, adjust=True):
    """
    Download price data for list or str of tickers.
    If adjust=True, uses auto_adjust=True so returned 'Close' is already adjusted.
    Returns a DataFrame (index=dates, columns=tickers).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=adjust,
        progress=False,
        group_by="column",
        threads=True
    )
    if df.empty:
        raise ValueError("No data returned from yfinance. Check tickers/date range/network.")

    # With multiple tickers, columns are MultiIndex: (Field, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in set(df.columns.get_level_values(0)):
            raise KeyError(f"'Close' not found in columns: {sorted(set(df.columns.get_level_values(0)))}")
        prices = df["Close"].copy()
    else:
        # Single ticker: columns are single Index
        if "Close" not in df.columns:
            raise KeyError(f"'Close' not found in columns: {df.columns.tolist()}")
        prices = df[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Keep only columns for which data actually came back
    available = [t for t in tickers if t in prices.columns]
    if not available:
        raise ValueError("Requested tickers returned no usable 'Close' series.")
    prices = prices[available]

    # Drop all-NaN rows and forward-fill small gaps
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    return prices

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def sharpe_series(returns: pd.DataFrame, rf_daily=0.0) -> pd.Series:
    mu = returns.mean()
    vol = returns.std()
    return (mu - rf_daily) / vol

def annualized_perf(return_series: pd.Series) -> tuple[float, float, float]:
    """
    From daily return series, compute (annualized_return, annualized_vol, sharpe).
    """
    mu_ann = return_series.mean() * TRADING_DAYS
    vol_ann = return_series.std() * np.sqrt(TRADING_DAYS)
    sharpe = (mu_ann - RISK_FREE_ANNUAL) / vol_ann if vol_ann > 0 else np.nan
    return mu_ann, vol_ann, sharpe

def max_drawdown(return_series: pd.Series) -> float:
    cum = (1 + return_series).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()

def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.Series | pd.DataFrame:
    return (1 + returns).cumprod() - 1

def print_weights_table(weights: np.ndarray, columns: list[str], title="Weights"):
    wdf = pd.DataFrame({"Ticker": columns, "Weight": weights})
    wdf = wdf.sort_values("Weight", ascending=False).reset_index(drop=True)
    print(f"\n{title}:\n", wdf.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# =========================
# Optimization (Max Sharpe)
# =========================
def portfolio_stats(weights, mu, cov, rf_daily=0.0):
    port_mu = np.dot(weights, mu)
    port_var = float(np.dot(weights.T, np.dot(cov, weights)))
    port_vol = np.sqrt(port_var)
    sharpe = (port_mu - rf_daily) / port_vol if port_vol > 0 else -np.inf
    return port_mu, port_vol, sharpe

def solve_max_sharpe(mu, cov, rf_daily=0.0, bounds=(0,1)):
    n = len(mu)
    x0 = np.repeat(1/n, n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bnds = tuple([bounds] * n)

    def neg_sharpe(w):
        _, vol, sh = portfolio_stats(w, mu, cov, rf_daily)
        return -sh

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x

# =========================
# Main Flow
# =========================

# 1) Download DJIA prices and returns
prices = download_prices(DJIA_TICKERS, start=START, end=END, adjust=True)
retns  = daily_returns(prices)

# 2) Rank by daily Sharpe and pick top N
sr = sharpe_series(retns, rf_daily=RISK_FREE_DAILY)
top = sr.nlargest(TOP_N)
selected_tickers = top.index.tolist()
selected = retns[selected_tickers].copy()

print("\nTop assets by daily Sharpe (DJIA, {} to {}):".format(START, END))
print(top.round(4).to_string())

# 3) Plot cumulative returns of the selected assets
cum_sel = cumulative_returns(selected)
plt.figure(figsize=(10,6))
cum_sel.plot(ax=plt.gca(), legend=True)
plt.title(f"Cumulative Returns — Top {TOP_N} DJIA Stocks (Adj Close)")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# 4) Build covariance/means and solve tangency (max Sharpe) portfolio
mu = selected.mean()                 # daily expected returns
cov = selected.cov()                 # daily covariance
weights = solve_max_sharpe(mu.values, cov.values, rf_daily=RISK_FREE_DAILY, bounds=LONG_ONLY_BOUNDS)

print_weights_table(weights, selected.columns.tolist(), title="Max-Sharpe (Long-Only) Weights")

# 5) Portfolio time series and performance
port_daily = (selected * weights).sum(axis=1)
port_cum = cumulative_returns(port_daily)

plt.figure(figsize=(10,6))
plt.plot(port_cum, label="Optimized Portfolio")
plt.title("Cumulative Returns — Optimized (Max-Sharpe) Portfolio")
plt.xlabel("Date"); plt.ylabel("Cumulative Return")
plt.legend(); plt.tight_layout(); plt.show()

ann_mu, ann_vol, ann_sharpe = annualized_perf(port_daily)
mdd = max_drawdown(port_daily)

print("\nPortfolio Performance ({} to {}):".format(START, END))
print(f"Annualized Return:   {ann_mu: .4%}")
print(f"Annualized Volatility:{ann_vol: .4%}")
print(f"Sharpe Ratio:         {ann_sharpe: .3f}")
print(f"Max Drawdown:         {mdd: .2%}")

# 6) Hierarchical clustering on correlation distance (correct method)
corr = selected.corr()
dist = 1 - corr  # correlation distance in [0,2]
Z = linkage(squareform(dist, checks=False), method="average")

plt.figure(figsize=(10,6))
dendrogram(Z, labels=corr.columns, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram (Correlation Distance, Average Linkage)")
plt.tight_layout()
plt.show()

# 7) Benchmark comparison (SPY and DIA)
bench1_prices = download_prices(BENCH1, start="2019-07-01", end=END, adjust=True).squeeze()
bench2_prices = download_prices(BENCH2, start="2019-07-01", end=END, adjust=True).squeeze()

def series_to_ann(ser: pd.Series) -> tuple[float, float]:
    rets = ser.pct_change().dropna()
    mu, vol, sh = annualized_perf(rets)
    return mu, sh

b1_mu, b1_sh = series_to_ann(bench1_prices)
b2_mu, b2_sh = series_to_ann(bench2_prices)

print("\nBenchmark Performance (2019-07-01 to {}):".format(END))
print(f"{BENCH1} Annualized Return: {b1_mu: .4%} | Sharpe: {b1_sh: .3f}")
print(f"{BENCH2} Annualized Return: {b2_mu: .4%} | Sharpe: {b2_sh: .3f}")

# Plot portfolio vs SPY for same period (rebuild portfolio series over that window for apples-to-apples)
common_start = "2019-07-01"
sel_prices_for_period = download_prices(selected_tickers, start=common_start, end=END, adjust=True)
sel_rets_for_period = sel_prices_for_period.pct_change().dropna()
port_daily_same = (sel_rets_for_period * weights).sum(axis=1)

cum_port_same = cumulative_returns(port_daily_same)
cum_bench1 = cumulative_returns(bench1_prices.pct_change().dropna())

plt.figure(figsize=(10,6))
plt.plot(cum_port_same, label="Portfolio (Max-Sharpe)")
plt.plot(cum_bench1, label=BENCH1)
plt.title(f"Cumulative Returns Since {common_start}: Portfolio vs {BENCH1}")
plt.xlabel("Date"); plt.ylabel("Cumulative Return")
plt.legend(); plt.tight_layout(); plt.show()
