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


train_img=train_img.astype(np.float64)/255.0
val_img=val_img.astype(np.float64)/255.0

import mynn
from mynn import layer,activation,models
from tqdm import tqdm

model = models.Model([
    layer.Dense(784,128, activation.Relu()),
    layer.Dense(128,50, activation.Relu()),
    layer.Dense(50,10, None)
], 10, cls_activation=activation.Softmax())

EPOCHS=100
batch_size=20
for epoch in range(EPOCHS):
    t_num=len(train_label)
    tqdm_bar=tqdm(range(int(t_num/batch_size)))
    losses=list()
    for batch_index in tqdm_bar:
        imgs=train_img[batch_index*batch_size:(batch_index+1)*batch_size]
        labels = train_label[batch_index*batch_size:(batch_index+1)*batch_size]
        if len(imgs)<10:
            # print(len(imgs))
            pass
        loss = model.train(imgs, labels)
        losses.append(loss)
        # TODO 
        tqdm_bar.set_description("epoch:{: ^4} avg_loss:{:.4f} ".format(
                epoch+1,
                float(sum(losses))/len(losses),
            ))
    pass
    
    
    total_correct = 0
    count_num = 0
    v_num=len(val_label)
    tqdm_bar=tqdm(range(int(v_num/batch_size)))
    for batch_index in tqdm_bar:
        imgs=train_img[batch_index*batch_size:(batch_index+1)*batch_size]
        labels = train_label[batch_index*batch_size:(batch_index+1)*batch_size]
        
        pred = model.predict(imgs)
        crtn = np.sum(np.equal(pred,labels))
        total_correct+=crtn
        count_num+=batch_size
        pass
        tqdm_bar.set_description("epoch:{: ^4}  accuracy:{:.4f}".format(
                epoch+1,
                float(total_correct)/count_num,
            ))
    # print('epoch:{}  test acc:{}'.format(epoch, round(correct_num/(5000),4)))
        
