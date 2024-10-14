from utilities import (xavier, sigmoid, mean_squared_error_derivative, sigmoid_derivative,
                       scale, mean_squared_error, accuracy, plot, softmax)
import numpy as np
from tqdm import tqdm


class TwoLayersNeural:
    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = n_hidden
        self.W1 = xavier(n_features, n_hidden)
        self.W2 = xavier(n_hidden, n_classes)
        self.b1 = xavier(1, n_hidden)
        self.b2 = xavier(1, n_classes)
        # self.b1 = np.zeros((1, n_hidden))  # Bias initialized to zero
        # self.b2 = np.zeros((1, n_classes))
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        # self.a2 = softmax(self.z2)
        return self.a2

    def backprop(self, X, y, alpha):
        # calculate the error
        a2 = self.a2
        error = mean_squared_error_derivative(a2, y)

        sigmoid_grad = sigmoid_derivative(self.z2)
        delta = error * sigmoid_grad

        # Calculate gradients for first layer
        weight2_grad = np.dot(self.a1.T, delta) / X.shape[0]
        bias2_grad = np.mean(delta, axis=0, keepdims=True)

        # Propagate error to the first layer
        error_hidden = np.dot(delta, self.W2.T) * sigmoid_derivative(self.z1)

        # Calculate gradients for hidden layer W1 and b1
        weight1_grad = np.dot(X.T, error_hidden) / X.shape[0]
        bias1_grad = np.mean(error_hidden, axis=0, keepdims=True)

        # Apply gradients to update
        self.W2 -= alpha * weight2_grad
        self.b2 -= alpha * bias2_grad
        self.W1 -= alpha * weight1_grad
        self.b1 -= alpha * bias1_grad

    def train(self, X, y, X_test, y_test, epochs=10, batch_size=100, learning_rate=.5):
        n_samples = X.shape[0]
        train_loss_history = []
        test_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        interval = int(epochs * 0.2)  # Printing interval every 20% of epochs

        # for epoch in tqdm(range(epochs), desc="Epochs"):
        for epoch in range(epochs):
            # output = None
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # for i in tqdm(range(0, n_samples, batch_size), desc="Batches", leave=False):
            for i in range(0, n_samples, batch_size):
                X_batch = scale(X_shuffled[i:i + batch_size])
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Backpropagation step
                self.backprop(X_batch, y_batch, learning_rate)

            test_acc = accuracy(self, scale(X_test), y_test)
            test_accuracy_history.append(test_acc)

        # plot(test_loss_history, test_accuracy_history, "Test")
        return test_accuracy_history
