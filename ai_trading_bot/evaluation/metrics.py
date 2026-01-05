import numpy as np
import pandas as pd

def cagr(equity, periods_per_year=252):
    total_periods = len(equity)
    if total_periods == 0:
        return 0
    return (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / total_periods) - 1

def volatility(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

def max_drawdown(equity):
    cumulative_max = equity.cummax()
    drawdown = (equity - cumulative_max) / cumulative_max
    return drawdown.min()
