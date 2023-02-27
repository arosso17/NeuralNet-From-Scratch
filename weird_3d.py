import numpy as np
from NeuralNet import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('TKAgg')


def generate_3D_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5
    n = int((stop - start) / (interval))
    n = n * n  # since its two-dimensional we create an nxn grid

    x_points = np.arange(start, stop, interval)
    y_points = np.arange(start, stop, interval)
    samples = []
    labels = []
    for x in x_points:
        for y in y_points:
            # set the class
            z = 1
            if x < 3 and y > -2:
                z = 0
            if 1 > x > -1 and 7 < y < 10:
                z = 1

            samples.append([[x, y]])
            labels.append([[z]])

    return np.array(samples), np.array(labels)


if __name__ == "__main__":
    NN = NeuralNet((2, 100, 1))
    x_train, y_train = generate_3D_data()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    x, y = x_train.T
    ax1.scatter(x, y, y_train, color='red')

    NN.train(x_train, y_train, max_epochs=500, learning_rate=0.05)     # a (2, 100, 1) NN gets 100% after 500 with 0.05
    threshold = 0.5

    guesses = NN.predict(x_train)
    c = np.array([x[0][i * int(len(x[0]) ** (1 / 2)): (i + 1) * int(len(x[0]) ** (1 / 2))] for i in range(int(len(x[0]) // len(x[0]) ** (1 / 2)))])
    d = np.array([y[0][i * int(len(x[0]) ** (1 / 2)): (i + 1) * int(len(x[0]) ** (1 / 2))] for i in range(int(len(y[0]) // len(x[0]) ** (1 / 2)))])
    e = []
    temp = []
    for g in guesses:
        temp.append(g[0][0])
        if len(temp) == len(x[0]) ** (1 / 2):
            e.append(temp)
            temp = []
    e = np.array(e)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title('"Regression" surface')
    ax2.plot_surface(c, d, e, color='green')
    guesses = [1 if guess > threshold else 0 for guess in guesses]
    loss = NN.loss(y_train, guesses)
    print("loss on the test set is", loss)
    s = sum([1 if guesses[i] == y_train[i] else 0 for i in range(len(guesses))]) / len(guesses)
    print("Accuracy is", s)
    ax1.set_title("Predictions")
    ax1.scatter(x, y, guesses, color='blue')
    plt.show()
