import numpy as np
from numpy import ndarray

class Loss:
    def loss(self, predicted: ndarray, target: ndarray) -> float:
        raise NotImplementedError

    def grad(self, predicted: ndarray, target: ndarray) -> ndarray:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: ndarray, target: ndarray) -> float:
        return np.sum((predicted - target) ** 2)

    def grad(self, predicted: ndarray, target: ndarray) -> ndarray:
        return 2 * (predicted - target)