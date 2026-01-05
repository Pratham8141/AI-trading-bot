import pandas as pd

def buy_and_hold(df, initial_capital=100000):
    df = df.copy()
    df["position"] = 1

    df["strategy_return"] = df["simple_return"]
    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_capital

    return df

def moving_average_crossover(df, short_window=20, long_window=50, initial_capital=100000):
    df = df.copy()

    df["signal"] = 0
    df.loc[df["sma_20"] > df["ema_20"], "signal"] = 1
    df.loc[df["sma_20"] <= df["ema_20"], "signal"] = 0

    df["position"] = df["signal"].diff().fillna(0)

    df["strategy_return"] = df["signal"].shift(1) * df["simple_return"]
    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_capital

    return df
