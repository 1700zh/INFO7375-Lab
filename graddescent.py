class GradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.params.weights -= self.learning_rate * layer.dW
        layer.params.bias -= self.learning_rate * layer.db
