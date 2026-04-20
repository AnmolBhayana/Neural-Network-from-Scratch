"""
Neural Network from Scratch
============================
A fully connected neural network implemented using only Python and NumPy.
No TensorFlow, no PyTorch — every component built manually.

Components:
- Forward propagation
- Backpropagation (chain rule)
- Gradient descent
- Activation functions: ReLU, Sigmoid, Softmax
- Loss functions: Binary Cross-Entropy, Categorical Cross-Entropy
"""

import numpy as np


# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

def binary_cross_entropy(Y, A):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m
    return loss

def categorical_cross_entropy(Y, A):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A + 1e-8)) / m
    return loss


# ─────────────────────────────────────────────
# Neural Network Class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully connected neural network.

    Parameters
    ----------
    layer_dims : list
        Number of units per layer including input.
        e.g. [784, 128, 64, 10] means:
        - 784 input features
        - 2 hidden layers (128, 64 units)
        - 10 output classes
    hidden_activation : str
        Activation for hidden layers: 'relu' or 'sigmoid'
    output_activation : str
        Activation for output layer: 'sigmoid' or 'softmax'
    learning_rate : float
        Step size for gradient descent
    """

    def __init__(self, layer_dims, hidden_activation='relu',
                 output_activation='sigmoid', learning_rate=0.01):
        self.layer_dims = layer_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.params = {}
        self.cache = {}
        self.loss_history = []
        self._init_params()

    def _init_params(self):
        """He initialisation for ReLU, Xavier for Sigmoid."""
        np.random.seed(42)
        L = len(self.layer_dims)
        for l in range(1, L):
            if self.hidden_activation == 'relu':
                scale = np.sqrt(2.0 / self.layer_dims[l - 1])
            else:
                scale = np.sqrt(1.0 / self.layer_dims[l - 1])
            self.params[f'W{l}'] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]) * scale
            self.params[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def _activate(self, Z, layer_index):
        """Apply appropriate activation based on layer position."""
        L = len(self.layer_dims) - 1
        if layer_index == L:
            if self.output_activation == 'softmax':
                return softmax(Z)
            return sigmoid(Z)
        if self.hidden_activation == 'relu':
            return relu(Z)
        return sigmoid(Z)

    def _activate_derivative(self, Z, layer_index):
        """Derivative of activation for backprop."""
        L = len(self.layer_dims) - 1
        if layer_index == L:
            return sigmoid_derivative(Z)
        if self.hidden_activation == 'relu':
            return relu_derivative(Z)
        return sigmoid_derivative(Z)

    # ─────────────────────────────────────────
    # Forward Propagation
    # ─────────────────────────────────────────

    def forward(self, X):
        """
        Forward pass through all layers.
        Stores Z and A values in cache for backprop.
        """
        self.cache['A0'] = X
        L = len(self.layer_dims) - 1
        A = X

        for l in range(1, L + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            Z = np.dot(W, A) + b
            A = self._activate(Z, l)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        return A

    # ─────────────────────────────────────────
    # Backpropagation
    # ─────────────────────────────────────────

    def backward(self, Y):
        """
        Backward pass — compute gradients for all layers
        using the chain rule.
        """
        grads = {}
        L = len(self.layer_dims) - 1
        m = Y.shape[1]

        # Output layer gradient
        AL = self.cache[f'A{L}']
        dAL = -(Y / (AL + 1e-8)) + (1 - Y) / (1 - AL + 1e-8)

        dZ = dAL * self._activate_derivative(self.cache[f'Z{L}'], L)
        grads[f'dW{L}'] = np.dot(dZ, self.cache[f'A{L-1}'].T) / m
        grads[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True) / m
        grads[f'dA{L-1}'] = np.dot(self.params[f'W{L}'].T, dZ)

        # Hidden layers — propagate gradient backwards
        for l in reversed(range(1, L)):
            dZ = grads[f'dA{l}'] * self._activate_derivative(self.cache[f'Z{l}'], l)
            grads[f'dW{l}'] = np.dot(dZ, self.cache[f'A{l-1}'].T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            grads[f'dA{l-1}'] = np.dot(self.params[f'W{l}'].T, dZ)

        return grads

    # ─────────────────────────────────────────
    # Gradient Descent — Weight Update
    # ─────────────────────────────────────────

    def update_params(self, grads):
        """Update weights and biases using gradient descent."""
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            self.params[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.params[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    # ─────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────

    def train(self, X, Y, epochs=1000, print_every=100):
        """
        Full training loop:
        forward → compute loss → backward → update weights
        """
        for epoch in range(epochs):
            AL = self.forward(X)

            if self.output_activation == 'softmax':
                loss = categorical_cross_entropy(Y, AL)
            else:
                loss = binary_cross_entropy(Y, AL)

            grads = self.backward(Y)
            self.update_params(grads)
            self.loss_history.append(loss)

            if epoch % print_every == 0:
                preds = self.predict(X)
                acc = self.accuracy(preds, Y)
                print(f"Epoch {epoch:>5} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

    # ─────────────────────────────────────────
    # Prediction & Evaluation
    # ─────────────────────────────────────────

    def predict(self, X):
        AL = self.forward(X)
        if self.output_activation == 'softmax':
            return np.argmax(AL, axis=0)
        return (AL >= 0.5).astype(int)

    def accuracy(self, predictions, Y):
        if self.output_activation == 'softmax':
            true_labels = np.argmax(Y, axis=0)
        else:
            true_labels = Y.flatten()
        return np.mean(predictions.flatten() == true_labels)
