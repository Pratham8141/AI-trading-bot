import matplotlib.pyplot as plt
import pandas as pd

def plot_equity(df, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(df["equity"])
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_drawdown(df, title, filename):
    equity = df["equity"]
    drawdown = (equity - equity.cummax()) / equity.cummax()

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, color="red")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
