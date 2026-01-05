import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from strategies.baselines import buy_and_hold, moving_average_crossover
from evaluation.plots import plot_equity, plot_drawdown

DATA_DIR = PROJECT_ROOT / "data" / "processed"

if __name__ == "__main__":
    for file in DATA_DIR.glob("*.csv"):
        print(f"Plotting for {file.name}")
        df = pd.read_csv(file)

        bh = buy_and_hold(df)
        mac = moving_average_crossover(df)

        plot_equity(bh, f"{file.name} — Buy & Hold")
        plot_drawdown(bh, f"{file.name} — Buy & Hold Drawdown")

        plot_equity(mac, f"{file.name} — MA Crossover")
        plot_drawdown(mac, f"{file.name} — MA Crossover Drawdown")
