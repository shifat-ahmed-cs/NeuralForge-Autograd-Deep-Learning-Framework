"""
Unit tests for NeuralForge autograd gradient correctness.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.autograd import Value

class TestNeuralForge(unittest.TestCase):
    def test_gradient_arithmetic(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x  # z = x*y + x -> dz/dx = y + 1 = 4, dz/dy = x = 2
        z.backward()
        self.assertEqual(z.data, 8.0)
        self.assertEqual(x.grad, 4.0)
        self.assertEqual(y.grad, 2.0)

    def test_relu(self):
        a = Value(-5.0)
        b = a.relu()
        b.backward()
        self.assertEqual(b.data, 0.0)
        self.assertEqual(a.grad, 0.0)

if __name__ == "__main__":
    unittest.main()
