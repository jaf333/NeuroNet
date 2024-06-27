import numpy as np
from numpy import ndarray

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs: ndarray) -> ndarray:
        raise NotImplementedError

    def backward(self, grad: ndarray) -> ndarray:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: ndarray) -> ndarray:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: ndarray) -> ndarray:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T