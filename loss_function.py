import numpy as np

class LossFunction:
    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Adding epsilon to prevent log(0)
        return loss

    @staticmethod
    def cross_entropy_loss_derivative(y_pred, y_true):
        m = y_true.shape[0]
        grad = - (y_true / (y_pred + 1e-9)) / m  # Adding epsilon to prevent division by zero
        return grad
