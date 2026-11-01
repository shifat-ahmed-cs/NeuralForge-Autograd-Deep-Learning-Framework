"""
Demonstration of NeuralForge: Autograd Graph Backprop and MLP Training on Non-linear XOR Problem.
"""
import os
import sys
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.autograd import Value
from src.mlp import MLP

def main():
    print("=" * 65)
    print("      NEURALFORGE: AUTOGRAD ENGINE & DEEP LEARNING SUITE      ")
    print("=" * 65)

    # 1. Reverse-Mode Autograd Automatic Differentiation
    a = Value(3.0)
    b = Value(-4.0)
    c = a * b
    d = c + Value(20.0)
    loss = d.tanh()

    loss.backward()
    print("[+] Autograd Computational Graph Backpropagation:")
    print("    - Expression: loss = tanh(a * b + 20)")
    print(f"    - Forward Value = {loss.data:.4f}")
    print(f"    - dLoss/da = {a.grad:.4f}")
    print(f"    - dLoss/db = {b.grad:.4f}")

    # 2. MLP Training on Non-Linear XOR Classification: (x1, x2) -> y in {-1, 1}
    random.seed(42)
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)]
    ]
    ys = [Value(-1.0), Value(1.0), Value(1.0), Value(-1.0)]

    model = MLP(nin=2, nouts=[4, 1])
    print(f"\n[+] Training Multi-Layer Perceptron (2 -> 4 -> 1, Tanh Activation) on XOR:")

    for epoch in range(100):
        ypred = [model(x)[0] for x in xs]
        loss_epoch = sum(((yp - yt)**2 for yp, yt in zip(ypred, ys)), Value(0.0))

        for p in model.parameters():
            p.grad = 0.0
        loss_epoch.backward()

        learning_rate = 0.1
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if (epoch + 1) % 20 == 0:
            print(f"    - Epoch {epoch+1:3d} | MSE Loss: {loss_epoch.data:.4f}")

    print("\n[+] Final Predictions on XOR Dataset:")
    for x, y_true in zip(xs, ys):
        pred = model(x)[0].data
        print(f"    - Input ({x[0].data:.0f}, {x[1].data:.0f}) -> Target: {y_true.data:4.1f} | Prediction: {pred:6.3f}")
    print("=" * 65)

if __name__ == "__main__":
    main()