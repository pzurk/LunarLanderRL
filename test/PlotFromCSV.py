import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

CUSTOM_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

def plot_reward_curves(filepaths):
    plt.figure(figsize=(12, 6))
    for i, filepath in enumerate(filepaths):
        data = pd.read_csv(filepath)
        label = os.path.basename(filepath)
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        plt.plot(data["Episode"], data["Avg100 Reward"], label=f"{label} - Avg100", linewidth=2, color=color)
        plt.plot(data["Episode"], data["Total Reward"], label=f"{label} - Total reward", alpha=0.3, linestyle='--', color=color)
    plt.title("Porównanie nagród w epizodach")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_success_rate_rolling(filepaths, window=5000):
    plt.figure(figsize=(12, 6))
    for i, filepath in enumerate(filepaths):
        data = pd.read_csv(filepath)
        label = os.path.basename(filepath)

        success_rolling = []
        for j in range(len(data)):
            if j < window:
                success_rolling.append(None)
            else:
                window_slice = data.iloc[j - window:j]
                success_count = (window_slice["Total Reward"] > 200).sum()
                rate = success_count * 100 / window
                success_rolling.append(rate)

        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        plt.plot(data["Episode"], success_rolling, label=f"{label} - Success Rate", color=color)

    plt.title(f"Success Rate (okno {window} epizodów)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate [%]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    args = sys.argv[1:]
    show_rewards = "--rewards" in args
    show_success = "--success" in args

    filepaths = [arg for arg in args if not arg.startswith("--")]

    if not filepaths:
        sys.exit(1)

    for path in filepaths:
        if not os.path.isfile(path):
            sys.exit(1)

    if not show_rewards and not show_success:
        show_rewards = True
        show_success = True

    if show_rewards:
        plot_reward_curves(filepaths)

    if show_success:
        plot_success_rate_rolling(filepaths)

if __name__ == "__main__":
    main()