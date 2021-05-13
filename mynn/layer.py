import numpy as np
from .activation import Softmax
class Dense:
    def __init__(self,in_channel,out_channel, activation):
        self.W=np.random.randn(out_channel, in_channel) * 0.1
        self.b=np.random.randn(1,out_channel) * 0.1
        self.activation = activation
        self.x = None
        self.t=None
        self.dW=None
        self.db=None
        
    def forward(self, x):
        self.x = x
        self.t = np.dot( x, self.W.T) + self.b
        if self.activation:
            out = self.activation.forward( self.t )
        else:
            out = self.t
        return out

    def backward(self, dA):
        m = self.x.shape[0]
        if self.activation:
            dt = self.activation.backward(dA, self.t)
        else:
            dt=dA
        self.dW = np.dot(dt.T, self.x)
        self.db = np.sum(dt, axis=0, keepdims=True)/m
        dA_prev = np.dot(dt,self.W)
        return dA_prev
    
    def update(self,lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
        pass
