from utilities import xavier, sigmoid, mean_squared_error_derivative, sigmoid_derivative
import numpy as np

class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, 10)
        self.a = None
        self.z = None

    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.biases
        self.a = sigmoid(self.z)
        return self.a


    def backprop(self, X, y, alpha):
        # Calculating gradients for each of
        # your weights and biases.

        # Updating your weights and biases.

        #calculate the error
        # self.forward(X)
        error = mean_squared_error_derivative(self.a, y)

        #calculate gradients
        sigmoid_grad = sigmoid_derivative(self.z)
        delta = error * sigmoid_grad

        #update weigths ans biases
        weight_grad = np.dot(X.T, delta) / len(X)
        bias_grad = np.mean(delta, axis=0, keepdims=True)

        # Apply gradients to update
        self.weights -= alpha * weight_grad
        self.biases -= alpha * bias_grad