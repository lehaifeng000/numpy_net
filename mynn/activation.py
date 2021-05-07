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
        return 1/(1+np.exp(-x))
    
    def backward(self, dA, x):
        dX = np.array(dA, copy=True)
        dX[x<=0] = 0
        return dA * dX

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

        # dt = np.zeros((x.shape))
        # a = self.forward(x)
        # for i in range(a.shape[0]):
        #    dt[i] = a[i] - np.sum(a[i] * a, axis=0, keepdims=True)

        # dt=np.array([10,10],dtype=np.float64)
        # for i in range(10):
        #     for j in range(10):
        #         if i==j:
        #             dt[i][j]=

        # sig = self.forward(x)
        # return dA * sig * (1-sig)
        return y_pre - y_true