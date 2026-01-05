import sys
from pathlib import Path

# Add project root to Python path (FIX)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from features.indicators import sma, ema, rsi, macd, bollinger_bands

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def add_indicators(df):
    close = df["Close"]

    df["sma_20"] = sma(close, 20)
    df["ema_20"] = ema(close, 20)
    df["rsi_14"] = rsi(close, 14)

    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    bb_upper, bb_lower = bollinger_bands(close)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    for file in PROCESSED_DIR.glob("*.csv"):
        print(f"Adding indicators to {file.name}")
        df = pd.read_csv(file)
        df = add_indicators(df)
        df.to_csv(file, index=False)
