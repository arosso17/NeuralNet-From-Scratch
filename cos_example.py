from NeuralNet import *


def generate_cos(n, spread=1, bounds=10):
    x_train = []
    y_train = []
    for i in range(0, n):
        val = np.random.uniform(-bounds, bounds)
        x_train.append([[val]])
        y_train.append([[np.cos(val) * np.random.normal(1, 0.01 * spread)]])
    return np.array(x_train), np.array(y_train)


if __name__ == "__main__":
    NN = NeuralNet((1, 100, 10, 1))
    x_train, y_train = generate_cos(1000, spread=2, bounds=10)


    fig1, ax1 = plt.subplots()
    ax1.set_title("Cos Training Data")
    ax1.scatter(x_train, y_train)

    NN.train(x_train, y_train, max_epochs=100, learning_rate=0.01)

    x_test, y_test = generate_cos(100, spread=2, bounds=10)

    guesses = NN.predict(x_test)
    loss = NN.loss(y_test, guesses)
    print("loss on the test set is", loss)

    fig2, ax2 = plt.subplots()
    plt.title("Cos Predictions")
    ax2.scatter(x_test, guesses)
    plt.show()

