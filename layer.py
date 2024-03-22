from parameter import Parameters

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.params = Parameters()
        self.params.initialize(input_dim, output_dim)
        self.activation = activation
        self.Z = None
        self.A_prev = None
        self.dW = None
        self.db = None
