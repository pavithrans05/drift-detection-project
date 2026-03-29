import numpy as np


class PageHinkley:
    def __init__(self, delta=0.01, lambda_=50):
        self.delta = delta
        self.lambda_ = lambda_
        self.mean = 0
        self.cum_sum = 0

    def update(self, x):
        self.mean += (x - self.mean)
        self.cum_sum += x - self.mean - self.delta

        if self.cum_sum > self.lambda_:
            self.cum_sum = 0
            return True

        return False


def compute_ph_scores(windows):
    """
    Convert window → scalar signal (mean)
    """
    return np.array([np.mean(w) for w in windows])