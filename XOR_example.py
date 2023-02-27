from NeuralNet import *


def generate_xor(n, spread=1):
    x_train = []
    y_train = []
    for i in range(0, n):
        x_train.append([[np.random.normal(0, 0.1 * spread), np.random.normal(0, 0.1 * spread)]])
        y_train.append([[0]])
        x_train.append([[np.random.normal(1, 0.1 * spread), np.random.normal(0, 0.1 * spread)]])
        y_train.append([[1]])
        x_train.append([[np.random.normal(0, 0.1 * spread), np.random.normal(1, 0.1 * spread)]])
        y_train.append([[1]])
        x_train.append([[np.random.normal(1, 0.1 * spread), np.random.normal(1, 0.1 * spread)]])
        y_train.append([[0]])
    return np.array(x_train), np.array(y_train)


if __name__ == "__main__":
    NN = NeuralNet((2, 4, 1))
    x_train, y_train = generate_xor(100, spread=2)

    green = []
    red = []
    for i in range(len(x_train)):
        if y_train[i][0] == 1:
            green.append(x_train[i][0])
        else:
            red.append(x_train[i][0])
    fig1, ax1 = plt.subplots()
    ax1.set_title("XOR Training Data")
    if red:
        ax1.scatter(*zip(*red), color='red')
    if green:
        ax1.scatter(*zip(*green), color='green')

    NN.train(x_train, y_train, max_epochs=100, learning_rate=0.05)

    x_test, y_test = generate_xor(100, spread=2)

    guesses = NN.predict(x_test)
    loss = NN.loss(y_test, guesses)
    print("loss on the test set is", loss)

    green = []
    blue = []
    red = []
    yellow = []
    # print(guesses)
    guesses = np.around(guesses)
    # print(guesses)
    for i in range(len(guesses)):
        if y_test[i][0] == 1:
            if guesses[i][0] == 1:
                green.append(x_test[i][0])
            else:
                blue.append(x_test[i][0])
        else:
            if guesses[i][0] == 1:
                yellow.append(x_test[i][0])
            else:
                red.append(x_test[i][0])

    fig2, ax2 = plt.subplots()
    plt.title("XOR Classification")
    if red:
        ax2.scatter(*zip(*red), color='red')
    if green:
        ax2.scatter(*zip(*green), color='green')
    if blue:
        ax2.scatter(*zip(*blue), color='blue')
    if yellow:
        ax2.scatter(*zip(*yellow), color='yellow')
    plt.show()

    # print(NN.predict([[0, 0]]))  # 0
    # print(NN.predict([[1, 0]]))  # 1
    # print(NN.predict([[0, 1]]))  # 1
    # print(NN.predict([[1, 1]]))  # 0

