import numpy as np

class Regularization:
    @staticmethod
    def l2_loss(weights, lambda_val):
        l2_loss = 0.5 * lambda_val * np.sum(np.square(weights))
        return l2_loss
