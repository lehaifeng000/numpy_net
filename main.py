import numpy as np
from pathlib import Path
data_dir='data'
file_paths={
    'train_img':'train-images-idx3-ubyte'
    ,'train_label':'train-labels-idx1-ubyte'
    ,'val_img':'t10k-images-idx3-ubyte'
    ,'val_label':'t10k-labels-idx1-ubyte'}

train_img=np.fromfile(Path(data_dir, file_paths['train_img']).open(), dtype=np.uint8)[16:].reshape((60000,-1))
train_label=np.fromfile(Path(data_dir, file_paths['train_label']).open(), dtype=np.uint8)[8:].reshape((60000,))

val_img=np.fromfile(Path(data_dir, file_paths['val_img']).open(), dtype=np.uint8)[16:].reshape((10000,-1))
val_label=np.fromfile(Path(data_dir, file_paths['val_label']).open(), dtype=np.uint8)[8:].reshape((10000,))


train_img=train_img.astype(np.float)/255.0
val_img=val_img.astype(np.float)/255.0

import mynn
from mynn import layer,activation

model = [
    layer.Dense(784,128, activation.Sigmoid()),
    layer.Dense(128,10, activation.Sigmoid()),
    layer.Dense(10,10, activation.Sigmoid())
]

EPOCHS=10

for epoch in range(EPOCHS):
    for batch_index in range(60000):
        img=train_img[batch_index]
        x=img[:,None]
        for l in model:
            x=l.forward(x)
        # TODO 
        pass
        
