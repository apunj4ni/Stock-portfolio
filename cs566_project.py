#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:43:43 2024

@author: aryanpunjani
"""
#%%retrieving data
import yfinance as yf
import pandas as pd

# List of stock tickers (S&P 500 or DJIA)
#sp500_tickers = [...]  # You can retrieve S&P 500 tickers as a list or use a subset for quicker testing
djia_tickers = djia_tickers = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]


# Function to download stock data
def download_data(tickers, start_date="2015-01-01", end_date="2023-01-01"):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Download data for both indices
#sp500_data = download_data(sp500_tickers)
djia_data = download_data(djia_tickers)
#%%daily returns
# Calculate daily returns
#sp500_returns = sp500_data.pct_change().dropna()
djia_returns = djia_data.pct_change().dropna()
#%%sharpe ratio
import numpy as np

# Define a risk-free rate (annualized; adjust based on your preference)
risk_free_rate = 0.01 / 252  # assuming 1% annual rate, converted to daily

# Calculate mean returns and volatility (standard deviation of returns)
def calculate_sharpe(returns, risk_free_rate):
    mean_returns = returns.mean()
    volatilities = returns.std()
    sharpe_ratios = (mean_returns - risk_free_rate) / volatilities
    return sharpe_ratios

# Calculate Sharpe ratios for each stock
#sp500_sharpe_ratios = calculate_sharpe(sp500_returns, risk_free_rate)
djia_sharpe_ratios = calculate_sharpe(djia_returns, risk_free_rate)

# Select top N assets based on Sharpe ratio
top_n = 10  # choose top 10 for a smaller, manageable portfolio
#selected_sp500 = sp500_sharpe_ratios.nlargest(top_n)
selected_djia = djia_sharpe_ratios.nlargest(top_n)

#print("Top S&P 500 assets by Sharpe ratio:\n", selected_sp500)
print("Top DJIA assets by Sharpe ratio:\n", selected_djia)
#%%plot cumulative retuturns on top assets
import matplotlib.pyplot as plt

# Plot cumulative returns for selected assets
def plot_cumulative_returns(data, title="Cumulative Returns"):
    cumulative_returns = (1 + data).cumprod() - 1
    cumulative_returns.plot(figsize=(10, 6))
    plt.title(title)
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.show()

# Plot for selected assets in S&P 500 and DJIA
#plot_cumulative_returns(sp500_returns[selected_sp500.index], title="Top S&P 500 Stocks")
plot_cumulative_returns(djia_returns[selected_djia.index], title="Top DJIA Stocks")
#%%define portfolio variance and return
#import numpy as np

# Calculate expected returns and covariance matrix for selected assets
selected_assets = djia_returns[selected_djia.index]  # or djia_returns[selected_djia.index]
expected_returns = selected_assets.mean()
cov_matrix = selected_assets.cov()

# Portfolio metrics calculation functions
def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))
#%%grid search
#minimize variance for target return
from scipy.optimize import minimize

# Objective function: minimize portfolio variance
def objective_function(weights):
    return portfolio_variance(weights, cov_matrix)

# Constraint: the sum of weights must be 1
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Bounds for weights (each weight is between 0 and 1)
bounds = [(0, 1) for _ in range(len(selected_assets.columns))]

# Target return constraint (optional, can be removed if you only want to minimize variance)
target_return = 0.001  # daily target return, adjust as needed

# Additional constraint for target return (optional)
constraints_target = [
    {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, expected_returns) - target_return},
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # weights sum to 1
]

# Run optimization
initial_guess = np.array([1/len(selected_assets.columns)] * len(selected_assets.columns))  # equal weights
optimized_results = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints_target)

# Optimized weights
optimal_weights = optimized_results.x
print("Optimal weights:", optimal_weights)
print("Expected portfolio return:", portfolio_return(optimal_weights, expected_returns))
print("Expected portfolio variance:", portfolio_variance(optimal_weights, cov_matrix))
#%%correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
corr_matrix = selected_assets.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix of Selected Assets")
plt.show()
#%%heirarchial clustering
#Using clustering, such as Hierarchical Clustering, can group stocks with similar behaviors, which helps in diversification. based on correlation
from scipy.cluster.hierarchy import linkage, dendrogram

# Perform hierarchical clustering
linked = linkage(corr_matrix, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=selected_assets.columns, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
#%%calculate returns# Calculate portfolio daily returns
portfolio_daily_returns = (selected_assets * optimal_weights).sum(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1

# Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label="Optimized Portfolio")
plt.title("Cumulative Returns of Optimized Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()
#%%performance metrics
# Sharpe ratio
annualized_return = portfolio_daily_returns.mean() * 252
annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate * 252) / annualized_volatility

# Maximum drawdown
rolling_max = (1 + portfolio_daily_returns).cumprod().expanding().max()
drawdown = (1 + portfolio_daily_returns).cumprod() / rolling_max - 1
max_drawdown = drawdown.min()

print("Annualized Return:", annualized_return)
print("Annualized Volatility:", annualized_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)

#%%code vs benchmark



# Step 1: Define a function to calculate annualized return and Sharpe ratio
def annualized_return(df):
    # Calculate total return over the period
    total_return = (df.iloc[-1] / df.iloc[0]) - 1
    # Calculate the annualized return (compounded)
    years = (df.index[-1] - df.index[0]).days / 365.25
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized

def sharpe_ratio(df, risk_free_rate=0.0):
    # Calculate daily returns
    daily_returns = df.pct_change().dropna()
    # Calculate annualized volatility (standard deviation)
    volatility = daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days
    # Calculate average return
    avg_return = daily_returns.mean() * 252  # Annualize the daily returns
    # Sharpe ratio formula
    return (avg_return - risk_free_rate) / volatility

# Step 2: Fetch the benchmark data (e.g., DJIA or SPY)
benchmark_ticker = 'SPY' 
benchmark_ticker2='DJIA' # You can change this to 'DJIA' or another ETF
benchmark_data = yf.download(benchmark_ticker, start="2019-07-01", end="2023-01-01")['Adj Close']
benchmark_data2 = yf.download(benchmark_ticker2, start="2019-07-01", end="2023-01-01")['Adj Close']

# Step 3: Calculate the annualized return and Sharpe ratio for the benchmark
benchmark_annualized_return = annualized_return(benchmark_data)
benchmark_sharpe_ratio = sharpe_ratio(benchmark_data)
benchmark_annualized_return2 = annualized_return(benchmark_data2)
benchmark_sharpe_ratio2 = sharpe_ratio(benchmark_data2)
# Step 4: Display Benchmark Performance
print("\nBenchmark1:SPY, Benchmark2:DJIA\n")
print(f"Benchmark1 Annualized Return: {benchmark_annualized_return:.4f}")
print(f"Benchmark1 Sharpe Ratio: {benchmark_sharpe_ratio:.4f}")
print(f"Benchmark2 Annualized Return: {benchmark_annualized_return2:.4f}")
print(f"Benchmark2 Sharpe Ratio: {benchmark_sharpe_ratio2:.4f}")
# Step 5: Compare the Portfolio to the Benchmark
portfolio_annualized_return = 0.252  # Portfolio annualized return from earlier
portfolio_sharpe_ratio = 0.982  # Portfolio Sharpe ratio from earlier

print("\nPortfolio vs Benchmark:")
print(f"Portfolio Annualized Return: {portfolio_annualized_return:.4f}")
print(f"Portfolio Sharpe Ratio: {portfolio_sharpe_ratio:.4f}")
print(f"Benchmark1 Annualized Return: {benchmark_annualized_return:.4f}")
print(f"Benchmark1 Sharpe Ratio: {benchmark_sharpe_ratio:.4f}")
print(f"Benchmark2 Annualized Return: {benchmark_annualized_return2:.4f}")
print(f"Benchmark2 Sharpe Ratio: {benchmark_sharpe_ratio2:.4f}")
