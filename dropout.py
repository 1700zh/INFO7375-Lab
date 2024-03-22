import numpy as np

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def apply(self, A, mode="train"):
        if mode == "train":
            self.mask = np.random.rand(*A.shape) > self.rate
            return A * self.mask
        else:
            return A * (1 - self.rate)
