import numpy as np


TRAINING_SET_X = np.array(
    [
        [0.78, 4.95, 1.19],
        [4.95, 1.19, 4.08],
        [1.19, 4.08, 0.8],
        [4.08, 0.8, 4.25],
        [0.8, 4.25, 0.22],
        [4.25, 0.22, 4.63],
        [0.22, 4.63, 1.48],
        [4.63, 1.48, 4.97],
        [1.48, 4.97, 0.53],
        [4.97, 0.53, 5.50]
    ]
)
TRAINING_SET_Y = np.array([4.08, 0.8, 4.25, 0.22, 4.63, 1.48, 4.97, 0.53, 5.50, 1.28])

TESTING_SET_X = np.array(
    [
        [0.53, 5.50, 1.28],
        [5.50, 1.28, 5.79]
    ]
)
TESTING_SET_Y = np.array([5.79, 0.44])


class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.normal(size=(3, 3)) * np.sqrt(1 / 3)
        self.B1 = np.random.normal(size=(3, 1)) * np.sqrt(1 / 3)
        self.W2 = np.random.normal(size=(1, 3))
        self.B2 = np.random.normal()

    @staticmethod
    def __relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def __relu_derivative(Z):
        return (Z > 0).astype(int)

    def __forward_propagation(self, X):
        Z1 = self.W1 @ X.T + self.B1
        A1 = self.__relu(Z1)
        Z2 = self.W2 @ A1 + self.B2
        A2 = self.__relu(Z2)
        return Z1, A1, Z2, A2

    def __backward_propagation(self, X, Z1, A1, A2, Y):
        num_of_training_sets = Y.shape[0]
        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / num_of_training_sets
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_training_sets
        dZ1 = self.W2.T @ dZ2 * self.__relu_derivative(Z1)
        dW1 = (dZ1 @ X) / num_of_training_sets
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_training_sets
        return dW1, db1, dW2, db2

    def __update_weights(self, dW1, dB1, dW2, dB2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.B1 = self.B1 - alpha * dB1
        self.W2 = self.W2 - alpha * dW2
        self.B2 = self.B2 - alpha * dB2

    def train(self, X, Y, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.__forward_propagation(X)
            dW1, dB1, dW2, dB2 = self.__backward_propagation(X, Z1, A1, A2, Y)
            self.__update_weights(dW1, dB1, dW2, dB2, alpha)

    def test_nn_accuracy(self, X):
        _, _, _, A2 = self.__forward_propagation(X)
        return A2


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.train(TRAINING_SET_X, TRAINING_SET_Y, 0.1, 250)
    predicted_nums = nn.test_nn_accuracy(TESTING_SET_X)
    print(predicted_nums)
