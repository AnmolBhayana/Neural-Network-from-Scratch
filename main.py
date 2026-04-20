"""
main.py — Run and evaluate the Neural Network from Scratch

Demonstrates:
1. Binary classification (Breast Cancer dataset)
2. Multi-class classification (Iris dataset)
3. Comparison against Scikit-learn baseline
4. Loss curve visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from neural_network import NeuralNetwork


def plot_loss_curve(loss_history, title="Training Loss Curve"):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='steelblue', linewidth=1.5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"Loss curve saved.")


# ─────────────────────────────────────────────
# DEMO 1: Binary Classification — Breast Cancer
# ─────────────────────────────────────────────

def demo_binary():
    print("\n" + "="*55)
    print("  DEMO 1: Binary Classification — Breast Cancer")
    print("="*55)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for our network: (features, samples)
    X_train_T = X_train.T
    X_test_T = X_test.T
    y_train_T = y_train.reshape(1, -1)
    y_test_T = y_test.reshape(1, -1)

    # Our Neural Network
    nn = NeuralNetwork(
        layer_dims=[30, 64, 32, 1],
        hidden_activation='relu',
        output_activation='sigmoid',
        learning_rate=0.005
    )
    nn.train(X_train_T, y_train_T, epochs=1000, print_every=200)

    test_preds = nn.predict(X_test_T)
    our_acc = nn.accuracy(test_preds, y_test_T)
    print(f"\n✅ Our Neural Network Test Accuracy:  {our_acc:.2%}")

    # Scikit-learn baseline
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"📊 Scikit-learn MLPClassifier Accuracy: {sk_acc:.2%}")

    plot_loss_curve(nn.loss_history, "Binary_Classification_Loss")


# ─────────────────────────────────────────────
# DEMO 2: Multi-class Classification — Iris
# ─────────────────────────────────────────────

def demo_multiclass():
    print("\n" + "="*55)
    print("  DEMO 2: Multi-class Classification — Iris")
    print("="*55)

    data = load_iris()
    X, y = data.data, data.target

    # One-hot encode labels
    def one_hot(y, num_classes):
        m = len(y)
        Y = np.zeros((num_classes, m))
        Y[y, np.arange(m)] = 1
        return Y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_T = X_train.T
    X_test_T = X_test.T
    y_train_oh = one_hot(y_train, 3)
    y_test_oh = one_hot(y_test, 3)

    # Our Neural Network
    nn = NeuralNetwork(
        layer_dims=[4, 16, 8, 3],
        hidden_activation='relu',
        output_activation='softmax',
        learning_rate=0.01
    )
    nn.train(X_train_T, y_train_oh, epochs=1000, print_every=200)

    test_preds = nn.predict(X_test_T)
    our_acc = nn.accuracy(test_preds, y_test_oh)
    print(f"\n✅ Our Neural Network Test Accuracy:  {our_acc:.2%}")

    # Scikit-learn baseline
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    sk_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"📊 Scikit-learn MLPClassifier Accuracy: {sk_acc:.2%}")

    plot_loss_curve(nn.loss_history, "Multiclass_Classification_Loss")


if __name__ == "__main__":
    demo_binary()
    demo_multiclass()
    print("\n✅ All demos completed.")
