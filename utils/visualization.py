import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


def plot_ks_drift(scores, title="KS Drift Score"):
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="KS Score")
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Drift Score")
    plt.grid()
    plt.legend()
    plt.show()


def plot_ks_with_true_drift(scores, true_drift_points, offset, title="KS vs True Drift"):
    """
    offset = ref_size (because KS starts later)
    """
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="KS Score")

    for p in true_drift_points:
        plt.axvline(x=p - offset, color='g', linestyle='--', label="True Drift" if p == true_drift_points[0] else "")

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.grid()
    plt.show()


def plot_ks_with_adwin(scores, adwin_points, offset, title="KS + ADWIN"):
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="KS Score")

    for p in adwin_points:
        plt.axvline(x=p - offset, color='r', linestyle='--', label="ADWIN Drift" if p == adwin_points[0] else "")

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.grid()
    plt.show()


def plot_all(scores, true_points, adwin_points, offset):
    plt.figure(figsize=(14, 5))
    plt.plot(scores, label="KS Score")

    # True drift (green)
    for p in true_points:
        plt.axvline(x=p - offset, color='g', linestyle='--', alpha=0.7,
                    label="True Drift" if p == true_points[0] else "")

    # ADWIN (red)
    for p in adwin_points:
        plt.axvline(x=p - offset, color='r', linestyle=':',
                    label="ADWIN" if p == adwin_points[0] else "")

    plt.title("Baseline Drift Detection (KS + ADWIN + Ground Truth)")
    plt.xlabel("Time Index")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.grid()
    plt.show()