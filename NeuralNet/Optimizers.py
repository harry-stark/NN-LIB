import numpy as np
import cupy as cp
class Optimizer(object):
    def __init__(self,parameters):
        self.parameters = parameters
    
    def step(self): 
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.parameters:
            p.grad = 0.


class SGD(Optimizer):
    def __init__(self,parameters,lr=.001,weight_decay=0.0,momentum = .9):
        super().__init__(parameters)
        self.lr           = lr
        self.weight_decay = weight_decay
        self.momentum     = momentum
        self.velocity     = []
        for p in parameters:
            self.velocity.append(np.zeros_like(p.grad))

    def step(self):
        for p,v in zip(self.parameters,self.velocity):
            v = self.momentum*v+p.grad+self.weight_decay*p.data
            p.data=p.data-self.lr*v
