import numpy as np
from activation import Activation

class ForwardPropagation:
    @staticmethod
    def linear_forward(A_prev, W, b):
        Z = np.dot(A_prev, W) + b
        return Z

    @staticmethod
    def activation_forward(A_prev, W, b, activation):
        Z = ForwardPropagation.linear_forward(A_prev, W, b)
        if activation == "relu":
            return Activation.relu(Z), Z
        elif activation == "sigmoid":
            return Activation.sigmoid(Z), Z
