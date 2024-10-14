import numpy as np
from utils.utils import custom_uniform
from matplotlib import pyplot as plt

np.random.uniform = custom_uniform


def scale(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.max(matrix)


def xavier(n_in: int, n_out: int) -> np.ndarray:
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    sig = sigmoid(z)
    return sig * (1 - sig)


def softmax(z):
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=0)


def mean_squared_error(predicted, expected, squared=False):
    mse = np.mean((predicted - expected) ** 2)
    return np.sqrt(mse) if squared else mse


def mean_squared_error_derivative(predicted, expected):
    return 2 * (predicted - expected)


def get_batches(X, batch_size):
    end = X.shape[0]
    for ndx in range(0, end, batch_size):
        yield X[ndx:min(ndx + batch_size, end)]


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


def accuracy(model, X, y):
    predicted = np.argmax(model.forward(X), axis=1)
    expected = np.argmax(y, axis=1)
    return np.mean(predicted == expected)

