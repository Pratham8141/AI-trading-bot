import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from env.trading_env import TradingEnv

df = pd.read_csv("data/processed/NIFTY50.csv")

env = TradingEnv(df)

obs, _ = env.reset()
print("Initial observation shape:", obs.shape)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print("Action:", action, "Reward:", reward)
    if done:
        break
