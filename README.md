# 🧠 Neural Network from Scratch

A fully connected neural network built from the ground up using **pure Python and NumPy** — no TensorFlow, no PyTorch, no shortcuts. Every component including forward propagation, backpropagation, and gradient descent is implemented manually.

---

## 📌 Overview

Most ML practitioners use deep learning frameworks without understanding what's happening underneath. This project was built to develop a deep, fundamental understanding of how neural networks actually learn — by implementing every part from scratch.

---

## 🎯 What's Implemented

- ✅ Fully connected (dense) neural network with configurable layers
- ✅ Forward propagation
- ✅ Backpropagation (manual chain rule implementation)
- ✅ Gradient descent optimisation
- ✅ Activation functions — Sigmoid, ReLU, Softmax
- ✅ Loss functions — Binary Cross-Entropy, Mean Squared Error
- ✅ Tested on classification datasets
- ✅ Results validated against Scikit-learn baselines

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| Math & Arrays | NumPy |
| Visualisation | Matplotlib |
| Validation | Scikit-learn |

---

## 🏗️ Architecture

```
Input Layer  →  Hidden Layer(s)  →  Output Layer
     ↓                ↓                  ↓
  Features       ReLU Activation     Softmax/Sigmoid
                      ↓
              Backpropagation
                      ↓
            Weight Update (GD)
```

---

## ⚙️ How It Works

### Forward Pass
Each layer computes:
```
Z = W · X + b
A = activation(Z)
```

### Backpropagation
Gradients are computed layer by layer using the chain rule:
```
dL/dW = dL/dA · dA/dZ · dZ/dW
```

### Weight Update
```
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install numpy matplotlib scikit-learn
```

### Run
```bash
git clone https://github.com/AnmolBhayana/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch
python main.py
```

---

## 📊 Results

- Matched Scikit-learn MLPClassifier accuracy on test classification datasets
- Demonstrated correct gradient flow through manual backprop implementation
- Loss curve shows consistent convergence across training epochs

---

## 💡 Key Learnings

- How gradients actually flow backward through a network
- Why weight initialisation matters
- The relationship between learning rate and convergence
- Why frameworks like PyTorch exist — and what they're abstracting away

---

## 🔮 Future Improvements

- Add momentum and Adam optimiser
- Implement dropout regularisation
- Extend to convolutional layers (CNN from scratch)

---

## 👤 Author

**Anmol Bhayana**
[LinkedIn](https://linkedin.com/in/a-721a2) • [GitHub](https://github.com/AnmolBhayana)
