import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_cash=100000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash

        self.feature_cols = [
            "simple_return",
            "log_return",
            "sma_20",
            "ema_20",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
        ]

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols) + 2,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0
        return self._get_observation(), {}

    def _get_observation(self):
        features = self.df.loc[self.current_step, self.feature_cols].values.astype(
            np.float32
        )
        return np.concatenate(
            [
                features,
                np.array(
                    [self.position, self.cash / self.initial_cash],
                    dtype=np.float32,
                ),
            ]
        )

    def step(self, action):
        done = False
        reward = 0.0

        price = self.df.loc[self.current_step, "Close"]
        prev_value = self.cash + self.shares * price

        trade_penalty = 0.001

        if action == 1 and self.position == 0:
            self.shares = self.cash / price
            self.cash = 0.0
            self.position = 1
            reward -= trade_penalty

        elif action == 2 and self.position == 1:
            self.cash = self.shares * price
            self.shares = 0.0
            self.position = 0
            reward -= trade_penalty

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            done = True
            next_price = price
        else:
            next_price = self.df.loc[self.current_step, "Close"]

        new_value = self.cash + self.shares * next_price
        reward += (new_value - prev_value) / self.initial_cash

        return self._get_observation(), reward, done, False, {}

    def render(self):
        price = self.df.loc[self.current_step, "Close"]
        value = self.cash + self.shares * price
        print(
            f"Step: {self.current_step} | "
            f"Price: {price:.2f} | "
            f"Position: {self.position} | "
            f"Portfolio Value: {value:.2f}"
        )
