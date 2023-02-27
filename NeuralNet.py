from Layer import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')


class NeuralNet:
    def __init__(self, size=(1, 5, 2, 1), loss=mean_squared_error, max_epochs=50):
        self.loss = loss
        if loss == mean_squared_error:
            self.loss_prime = mean_squared_error_prime
        self.size = size
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(size) - 1)]
        self.error = None
        self.max_epochs = max_epochs

    def predict(self, inp):
        vals = inp
        for layer in self.layers:
            vals = layer.forward_propagation(vals)
        return vals

    def train(self, samples, lables, max_epochs, learning_rate):
        for ep in range(max_epochs):
            # le = (1 - 1.25 ** (ep - max_epochs)) * learning_rate
            # le = learning_rate * np.emath.logn(max_epochs - 1, max_epochs + 1 - ep)
            le = learning_rate * (1 - (ep/(max_epochs + 1)))
            for i in range(len(samples)):
                out = self.predict(samples[i])
                self.back_prop(lables[i], out, le)
            print(f"Epoch {ep + 1} / {max_epochs}")

    def back_prop(self, label, output, learning_rate):
        error = self.loss_prime(label, output)
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

    def __str__(self):
        b = [layer.bias for layer in self.layers]
        w = [layer.weights for layer in self.layers]
        return "{}\n{}".format(w, b)


