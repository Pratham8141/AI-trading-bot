import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

def preprocess(file_path):
    df = pd.read_csv(file_path)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert price columns to numeric (CRITICAL FIX)
    for col in PRICE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid prices
    df = df.dropna(subset=["Close"])

    # Returns
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Final cleanup
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file in RAW_DIR.glob("*.csv"):
        print(f"Processing {file.name}")
        df = preprocess(file)
        df.to_csv(PROCESSED_DIR / file.name, index=False)
