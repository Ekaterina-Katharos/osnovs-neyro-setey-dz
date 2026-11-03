import numpy as np
from exceptions import InvalidLayerSizeError, MismatchedDataError, ModelNotTrainedError

class NeuralNetwork:
    """
    Полносвязная нейронная сеть (Многослойный перцептрон).
    """
    def __init__(self, layer_sizes: list, activation: str = 'sigmoid'):
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Сеть должна содержать минимум 2 слоя.")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоев должны быть положительными.")
        if activation not in ['sigmoid', 'relu']:
            raise ValueError("Доступные активации: 'sigmoid', 'relu'.")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.is_trained = False
        self.loss_history = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, x, derivative=False):
        if self.activation_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig) if derivative else sig
        elif self.activation_name == 'relu':
            return np.where(x > 0, 1, 0) if derivative else np.maximum(0, x)

    def forward(self, X):
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(f"Ожидалось {self.layer_sizes[0]} признаков, получено {X.shape[1]}.")
        
        self.activations = [X]
        self.z_values = []
        current_input = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = 1 / (1 + np.exp(-z)) if i == len(self.weights) - 1 else self._activation(z)
            self.activations.append(a)
            current_input = a
        return self.activations[-1]

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        output = self.activations[-1]
        error = output - y
        delta = error * (output * (1 - output))
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = (1/m) * np.dot(self.activations[i].T, delta)
            db = (1/m) * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation(self.z_values[i-1], derivative=True)
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        self.is_trained = True
        self.loss_history = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            epoch_loss = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                epoch_loss += np.mean((output - y_batch) ** 2) * len(X_batch)
                
            self.loss_history.append(epoch_loss / n_samples)

    def predict(self, X):
        if not self.is_trained:
            raise ModelNotTrainedError("Сначала обучите сеть!")
        output = self.forward(X)
        return (output > 0.5).astype(int) if output.shape[1] == 1 else output

    def save_weights(self, filepath: str):
        np.savez(filepath, *self.weights, *self.biases)

    def load_weights(self, filepath: str):
        try:
            data = np.load(filepath)
            self.weights = [data[f'arr_{i}'] for i in range(len(self.weights))]
            self.biases = [data[f'arr_{i+len(self.weights)}'] for i in range(len(self.biases))]
            self.is_trained = True
        except Exception as e:
            raise MismatchedDataError(f"Ошибка загрузки: {e}")