# src/lin_sgd/plotting.py
import matplotlib.pyplot as plt
from typing import Dict

def plot_training_history(history: Dict, title: str = "Training"):
    steps = history["step"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.grid()

    axes[0].plot(steps, history["loss"])
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")

    axes[1].plot(steps, history["lr"])
    axes[1].set_xlabel("step"); axes[1].set_ylabel("lr"); axes[1].set_title("Learning rate")

    axes[2].plot(steps, history["eta"])
    axes[2].set_xlabel("step"); axes[2].set_ylabel("eta"); axes[2].set_title("Eta")

    fig.tight_layout()
    plt.savefig(f"{title}")
