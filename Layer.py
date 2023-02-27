from utils import *

class Layer:
    def __init__(self, in_size, out_size, activation=tanh):
        self.activation = activation
        if self.activation == sigmoid:
            self.activation_prime = sigmoid_prime
        elif self.activation == tanh:
            self.activation_prime = tanh_prime
        self.in_size = in_size
        self.error = None
        self.input = None
        self.output = None
        self.activated_out = None
        self.bias = np.random.rand(1, out_size) - 0.5
        self.weights = np.random.rand(in_size, out_size) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        self.activated_out = self.activation(self.output)
        return self.activated_out

    def backward_propagation(self, output_error, learning_rate):
        output_error = self.activation_prime(self.output) * output_error
        # print("e", output_error)
        # print("w", self.weights)
        # print("b", self.bias)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


