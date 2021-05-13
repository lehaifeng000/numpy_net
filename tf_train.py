import numpy as np
from pathlib import Path
data_dir='data'
file_paths={
    'train_img':'train-images-idx3-ubyte'
    ,'train_label':'train-labels-idx1-ubyte'
    ,'val_img':'t10k-images-idx3-ubyte'
    ,'val_label':'t10k-labels-idx1-ubyte'}

cont = np.fromfile(Path(data_dir, file_paths['train_img']).open(), dtype=np.uint8)[:16]

train_img=np.fromfile(Path(data_dir, file_paths['train_img']).open(), dtype=np.uint8)[16:].reshape((60000,-1))
train_label=np.fromfile(Path(data_dir, file_paths['train_label']).open(), dtype=np.uint8)[8:].reshape((60000,))

val_img=np.fromfile(Path(data_dir, file_paths['val_img']).open(), dtype=np.uint8)[16:].reshape((10000,-1))
val_label=np.fromfile(Path(data_dir, file_paths['val_label']).open(), dtype=np.uint8)[8:].reshape((10000,))

import tensorflow as tf

train_img=train_img.astype(np.float)/255.0
val_img=val_img.astype(np.float)/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(784,activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_img, train_label, epochs=5)


model.evaluate(val_img, val_label)


pass
