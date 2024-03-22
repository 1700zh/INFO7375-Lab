import numpy as np
from activation import Activation

class BackwardPropagation:
    @staticmethod
    def linear_backward(dZ, cache):
        A_prev, W, _ = cache
        m = A_prev.shape[0]
        dW = np.dot(A_prev.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, W.T)
        return dA_prev, dW, db

    @staticmethod
    def activation_backward(dA, cache, activation):
        Z = cache
        if activation == "relu":
            dZ = np.multiply(dA, Activation.relu_derivative(Z))
        return dZ
