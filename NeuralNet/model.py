import numpy as np
from .Datagen import DataGenerator
class Model():
    def __init__(self):
        self.computation_graph = []
        self.parameters = []

    def add(self, layer):
        self.computation_graph.append(layer)
        self.parameters += layer.getParams()

    def __innitializeNetwork(self):
        for f in self.computation_graph:
            if f.type == 'linear':
                weights, bias = f.getParams()
                weights.data = .01 * np.random.randn(weights.data.shape[0], weights.data.shape[1])
                bias.data = 0.

    def fit(self, data, target, batch_size, num_epochs, optimizer, loss_fn):
        loss_history = []
        self.__innitializeNetwork()
        data_gen = DataGenerator(data, target, batch_size)
        itr = 0
        for epoch in range(num_epochs):
            for X, Y in data_gen:
                optimizer.zeroGrad()
                for f in self.computation_graph: X = f.forward(X)
                loss = loss_fn.forward(X, Y)
                grad = loss_fn.backward()
                for f in self.computation_graph[::-1]: grad = f.backward(grad)
                loss_history += [loss]
                print("Loss at epoch = {} and iteration = {}: {}".format(epoch, itr, loss_history[-1]))
                itr += 1
                optimizer.step()

        return loss_history

    def predict(self, data):
        X = data
        for f in self.computation_graph: X = f.forward(X)
        return X
