import numpy as np


# Набір тренувальних даних (входи)
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
# Набір тренувальних даних (виходи)
TRAINING_SET_Y = np.array([4.08, 0.8, 4.25, 0.22, 4.63, 1.48, 4.97, 0.53, 5.50, 1.28])

# Набір тестових даних (входи)
TESTING_SET_X = np.array(
    [
        [0.53, 5.50, 1.28],
        [5.50, 1.28, 5.79]
    ]
)
# Набір тестових даних (виходи)
TESTING_SET_Y = np.array([5.79, 0.44])


class NeuralNetwork:
    # Ініціалізація ваг та зсувів
    def __init__(self):
        self.W1 = np.random.normal(size=(3, 3)) * np.sqrt(1 / 3)
        self.B1 = np.random.normal(size=(3, 1)) * np.sqrt(1 / 3)
        self.W2 = np.random.normal(size=(1, 3))
        self.B2 = np.random.normal()

    # Функція активації ReLU
    @staticmethod
    def __relu(Z):
        return np.maximum(Z, 0)

    # Похідна функції активації ReLU
    @staticmethod
    def __relu_derivative(Z):
        return (Z > 0).astype(int)

    # Пряме поширення
    def __forward_propagation(self, X):
        # Лінійна комбінація ваг та входів для першого шару
        Z1 = self.W1 @ X.T + self.B1
        # Активація першого шару
        A1 = self.__relu(Z1)
        # Лінійна комбінація ваг та виходів першого шару для другого шару
        Z2 = self.W2 @ A1 + self.B2
        # Активація другого шару
        A2 = self.__relu(Z2)
        return Z1, A1, Z2, A2

    # Зворотне поширення
    def __backward_propagation(self, X, Z1, A1, A2, Y):
        # Кількість тренувальних наборів
        num_of_training_sets = Y.shape[0]
        # Помилка виходу
        dZ2 = A2 - Y
        # Градієнт для ваг другого шару
        dW2 = (dZ2 @ A1.T) / num_of_training_sets
        # Градієнт для зсувів другого шару
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_training_sets
        # Помилка першого шару
        dZ1 = self.W2.T @ dZ2 * self.__relu_derivative(Z1)
        # Градієнт для ваг першого шару
        dW1 = (dZ1 @ X) / num_of_training_sets
        # Градієнт для зсуву першого шару
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_training_sets
        return dW1, db1, dW2, db2

    # Оновлення ваг та зсувів
    def __update_weights(self, dW1, dB1, dW2, dB2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.B1 = self.B1 - alpha * dB1
        self.W2 = self.W2 - alpha * dW2
        self.B2 = self.B2 - alpha * dB2

    # Тренування нейронної мережі
    def train(self, X, Y, alpha, iterations):
        for i in range(iterations):
            # Пряме поширення
            Z1, A1, Z2, A2 = self.__forward_propagation(X)
            # Зворотне поширення
            dW1, dB1, dW2, dB2 = self.__backward_propagation(X, Z1, A1, A2, Y)
            # Оновлення ваг та зсувів
            self.__update_weights(dW1, dB1, dW2, dB2, alpha)

    # Прогнозування наступного значення часового ряду
    def predict(self, X):
        _, _, _, A2 = self.__forward_propagation(X)
        return A2


if __name__ == '__main__':
    # Ініціалізація нейронної мережі
    nn = NeuralNetwork()
    # Тренування нейронної мережі
    nn.train(TRAINING_SET_X, TRAINING_SET_Y, 0.1, 250)
    # Тестування нейронної мережі
    predicted_nums = nn.predict(TESTING_SET_X)[0]
    # Виведення результатів
    print(f'Очікуване значення: {TESTING_SET_Y[0]}. Спрогнозоване значення: {predicted_nums[0]}.')
    print(f'Очікуване значення: {TESTING_SET_Y[1]}. Спрогнозоване значення: {predicted_nums[1]}.')
