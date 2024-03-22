import numpy as np

class Normalization:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.mean = None
        self.var = None

    def fit(self, Z):
        # Calculate mean and variance for the batch
        self.mean = np.mean(Z, axis=0)
        self.var = np.var(Z, axis=0)

        # Initialize gamma and beta if they are not set yet (first batch)
        if self.gamma is None:
            self.gamma = np.ones((1, Z.shape[1]))
        if self.beta is None:
            self.beta = np.zeros((1, Z.shape[1]))

    def apply(self, Z):
        # Normalize
        Z_norm = (Z - self.mean) / np.sqrt(self.var + self.epsilon)
        # Scale and shift
        out = self.gamma * Z_norm + self.beta
        return out

    def update_params(self, dgamma, dbeta):
        self.gamma -= self.learning_rate * dgamma
        self.beta -= self.learning_rate * dbeta
