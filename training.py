from loss_function import LossFunction


class Training:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def train(self, X_train, y_train, epochs, lambda_val=0.01):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.forward_pass(X_train)

            # Compute loss using LossFunction class
            loss = LossFunction.cross_entropy_loss(y_pred, y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Backward pass
            d_loss = LossFunction.cross_entropy_loss_derivative(y_pred, y_train)
            self.model.backward_pass(d_loss)
            self.update_params(lambda_val)

    def update_params(self, lambda_val):
        for layer in self.model.layers:
            dL2 = lambda_val * layer.params.weights
            layer.params.weights -= self.learning_rate * (layer.dW + dL2)
            layer.params.bias -= self.learning_rate * layer.db
