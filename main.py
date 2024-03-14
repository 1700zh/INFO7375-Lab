from model import Model
from loss_function import LossFunction
from training import Training

class main:
    def __init__(self, learning_rate=0.01, lambda_val=0.01):
        self.model = Model()
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val

    def add_layer(self, input_dim, output_dim, activation):
       
        self.model.add_layer(input_dim, output_dim, activation)

    def setup_and_train(self, X_train, y_train, epochs):
       
        trainer = Training(self.model, self.learning_rate)
        trainer.train(X_train, y_train, epochs, self.lambda_val)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.forward_pass(X_test)
        loss = LossFunction.cross_entropy_loss(y_pred, y_test)
        print(f"Test Loss: {loss}")


