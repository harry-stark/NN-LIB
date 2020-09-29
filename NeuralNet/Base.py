import numpy as np
import cupy as cp

class Tensor():
     def __init__(self,shape):
         self.data=np.ndarray(shape,np.float32)
         self.grad=np.ndarray(shape,np.float32)
         self.shape=shape

class Function():
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

    def getParams(self):
        return []