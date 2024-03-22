from foward_p import ForwardPropagation
from layer import Layer
from loss_function import LossFunction


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation):
        layer = Layer(input_dim, output_dim, activation)
        self.layers.append(layer)

    def forward_pass(self, X):
        A = X
        for layer in self.layers:
            A, _ = ForwardPropagation.activation_forward(A, layer.params.weights, layer.params.bias, layer.activation)
        return A

    def backward_pass(self, y_pred, y_true):
        error_signal = LossFunction.cross_entropy_loss_derivative(y_pred, y_true)
        for layer in reversed(self.model.layers):
            error_signal = layer.backward_pass(error_signal)
