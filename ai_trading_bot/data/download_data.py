import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data/raw")

def download(symbol, name):
    print(f"Downloading {name}...")
    df = yf.download(symbol, start="2018-01-01", interval="1d")
    df.reset_index(inplace=True)
    df.to_csv(RAW_DIR / f"{name}.csv", index=False)

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download("^NSEI", "NIFTY50")
    download("^GSPC", "SP500")
    download("BTC-USD", "BTCUSD")
