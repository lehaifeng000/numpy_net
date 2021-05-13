import numpy as np
from abc import abstractmethod

class Activation:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        return

    @abstractmethod
    def backward(self):
        return

class Relu(Activation):
    def __init__(self):
        pass
    
    def forward(self, x):
        # return np.maximum(0,x)
        a = np.maximum(0,x)
        b=x*(x>0)
        return x*(x>0)
        return 1/(1+np.exp(-x))
    
    def backward(self, dA, x):
        dX = np.array(dA, copy=True)
        dX[x<=0] = 0
        return  dX

class Sigmoid(Activation):
    def __init__(self):
        pass
    
    def forward(self, x):
        self.x=x
        return 1/(1+np.exp(-x))
    
    def backward(self,dA,x):
        sig = self.forward(x)
        return dA * sig * (1-sig)

class Softmax(Activation):
    def __init__(self):
        pass
    
    def forward(self, x):
        self.x=x
        exp = np.exp(x)
        out = exp / np.sum(exp,axis=1,keepdims=True)
        return out
    
    def backward(self,y_pre,y_true):

        return y_pre - y_true