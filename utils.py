import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.dot(1 / (1 + np.exp(-x)), (1 - (1 / (1 + np.exp(-x)))).T)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mean_squared_error_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

