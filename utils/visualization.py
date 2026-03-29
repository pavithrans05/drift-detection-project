import matplotlib.pyplot as plt
import os


def plot_drift(scores, boundaries, title="Drift Detection", filename=None):
    plt.figure(figsize=(12, 5))

    plt.plot(scores, label="Drift Score")

    for start, _ in boundaries:
        plt.axvline(x=start, linestyle='--', alpha=0.5)

    plt.title(title)
    plt.xlabel("Time (Window Index)")
    plt.ylabel("Drift Score")
    plt.legend()

    if filename:
        os.makedirs("outputs/plots", exist_ok=True)
        path = os.path.join("outputs/plots", filename)
        plt.savefig(path, bbox_inches="tight")
        print(f"Plot saved at: {path}")

    plt.show()