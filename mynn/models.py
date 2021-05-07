import numpy as np


def label2onehot(y,num_cluster):
    # onehot=np.zeros((y.shape[0],),dtype=np.int64)
    # onehot[y]=1
    eye=np.eye(num_cluster, dtype=np.int64)
    onehot = eye[y]
    return onehot

def cross_entropy(y_pre,y_true):
    loss = - (y_true * np.log(y_pre))
    loss = np.sum(loss)/loss.shape[0]
    # dA = - (y_true / y_pre)
    return loss #, dA


class Model:
    def __init__(self, layers, num_cluster,cls_activation, lr=1e-4):
        self.layers = layers
        self.num_cluster = num_cluster
        self.lr = lr 
        self.cls_activation = cls_activation

        pass

    def train(self, X, Y):
        out=X
        label = label2onehot(Y, self.num_cluster)
        for layer in self.layers:
            out=layer.forward(out)
        # compute loss
        out = self.cls_activation.forward(out)

        loss= cross_entropy(out, label)
        # print("loss:  ",loss)
        dA = self.cls_activation.backward(out, label)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        for layer in self.layers:
            layer.update(self.lr)

        return loss
    
    def predict(self, X):
        out=X
        for layer in self.layers:
            out=layer.forward(out)
        pred = np.argmax(out,axis=1)
        return pred

    

