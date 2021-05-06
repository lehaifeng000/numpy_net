import numpy as np

class Dense:
    def __init__(self,in_channel,out_channel, activation):
        self.W=np.random.randn(out_channel, in_channel) * 0.1
        self.b=np.random.randn(out_channel,1) * 0.1
        self.activation = activation
        self.x = None
        self.t=None
        self.dW=None
        self.db=None
        
    def forward(self, x):
        self.x = x
        t = np.dot( self.W, x) + self.b
        self.t=t
        out = self.activation.forward( t )
        return out

    def backward(self, dA):
        m = self.x.shape[1]
        dt = self.activation.backward(dA, self.t)
        selt.dW = np.dot(dt, self.x.T)/m
        selt.db = np.sum(dt, axis=1, keepdims=True)/m
        dA_prev = np.dot(self.W.T, dt)
        return dA_prev
