# NeuralForge: Deep Learning Autograd Engine & Neural Network Library from Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI: Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Domain: Deep Learning](https://img.shields.io/badge/domain-Neural%20Networks-red.svg)]()
[![Academic: BRACU CSE425](https://img.shields.io/badge/course-CSE%20425-orange.svg)]()

## 📌 Course Context: CSE 425 (Neural Network)
*Department of Computer Science and Engineering, BRAC University*  
**Core Principles:** Dynamic Computational Graphs, Reverse-Mode Automatic Differentiation (Autograd), Topological Backpropagation, Multi-Layer Perceptrons (MLP), and Gradient Descent Optimization.

---

## 🎯 Architecture & Autograd Engine

```
[Inputs] ---> (Weight Multiplication) ---> (Biases) ---> [Non-linear Activation: Tanh]
                                                                   |
                                                                   v
                                                             [Prediction]
                                                                   |
                                                                   v
                                                        [Loss Function: MSE]
                                                                   |
(Gradients via Reverse-Mode Topological Sort) <--------------------+
```

1. **Scalar Autograd Engine (`Value` class):**
   - Supports $+$, $-$, $*$, $/$, $**$ (power), $\text{ReLU}$, $\text{LeakyReLU}$, $\tanh$, and $\text{Sigmoid}$.
   - Automatic reverse-mode backpropagation building DAG topological sorts.
2. **Multi-Layer Perceptron (MLP):**
   - Composed of modular `Neuron`, `Layer`, and `MLP` architectures.
   - Solves non-linear XOR classification with convergence to $< 0.1$ MSE loss.

---

## 🚀 Quickstart & Testing
```bash
# Run demonstration
python main.py

# Run unit tests
python -m unittest discover -s tests
```
