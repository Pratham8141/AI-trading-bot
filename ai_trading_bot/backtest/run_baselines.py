import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from strategies.baselines import buy_and_hold, moving_average_crossover
from evaluation.metrics import cagr, sharpe_ratio, max_drawdown, volatility

DATA_DIR = PROJECT_ROOT / "data" / "processed"

def print_metrics(name, df):
    print(f" {name}")
    print("  CAGR:", round(cagr(df["equity"]) * 100, 2), "%")
    print("  Sharpe:", round(sharpe_ratio(df["strategy_return"]), 2))
    print("  Volatility:", round(volatility(df["strategy_return"]) * 100, 2), "%")
    print("  Max Drawdown:", round(max_drawdown(df["equity"]) * 100, 2), "%")

if __name__ == "__main__":
    for file in DATA_DIR.glob("*.csv"):
        print(f"\nAsset: {file.name}")
        df = pd.read_csv(file)

        bh = buy_and_hold(df)
        mac = moving_average_crossover(df)

        print_metrics("Buy & Hold", bh)
        print_metrics("MA Crossover", mac)
        print("-" * 50)
