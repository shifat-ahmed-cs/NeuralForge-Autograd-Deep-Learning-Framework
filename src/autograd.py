"""
Scalar/Tensor Autograd Engine with Dynamic Computational Graph and Reverse-Mode AD.
"""
import math
from typing import List, Tuple, Union

class Value:
    """Scalar node in dynamic computational graph supporting automatic backpropagation."""
    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: Union['Value', float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Union['Value', float]) -> 'Value':
        return self + other

    def __neg__(self) -> 'Value':
        return self * -1.0

    def __sub__(self, other: Union['Value', float]) -> 'Value':
        return self + (-other)

    def __rsub__(self, other: Union['Value', float]) -> 'Value':
        return other + (-self)

    def __mul__(self, other: Union['Value', float]) -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other: Union['Value', float]) -> 'Value':
        return self * other

    def __pow__(self, other: Union[int, float]) -> 'Value':
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self, alpha: float = 0.01) -> 'Value':
        out = Value(self.data if self.data > 0 else alpha * self.data, (self,), "LeakyReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> 'Value':
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Value':
        s = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.data))))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Topological sort reverse-mode backpropagation."""
        topo: List[Value] = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()