import numpy as np
from neuronet.train import train
from neuronet.neural_network import NeuralNet
from neuronet.layers import Linear
from neuronet.activations import Relu
from neuronet.optimizer import SGD

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=5),
    Relu(),
    Linear(input_size=5, output_size=2)
])

train(net, inputs, targets, num_epoch=1000, optimizer=SGD(lr=0.01))

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(f"Input: {x} -> Predicted: {predicted.round(2)} -> Target: {y}")