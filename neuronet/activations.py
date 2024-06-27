import numpy as np
from numpy import ndarray
from typing import Callable

Func = Callable[[ndarray], ndarray]

class Activation(Layer):
    def __init__(self, f: Func, f_prime: Func) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return self.f(self.inputs)

    def backward(self, grad: ndarray) -> ndarray:
        return self.f_prime(self.inputs) * grad

def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)

def relu_prime(x: ndarray) -> ndarray:
    return (x > 0).astype(x.dtype)

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)