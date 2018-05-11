import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import keras
from keras import backend as K
import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import *


def load():
    #f = h5py.File('data_nju2000_224x224.h5','r')    
    f = h5py.File('data_nju1500_nlpr500_224x224.h5','r')    
    #loaded_obj = pickle.load(f)
    #f.close()
    #X, y = loaded_obj
#data labels
    f.keys()
    x = f['x'][:]
    y = f['y'][:]
    val_x = f['val_x'][:]
    val_y = f['val_y'][:]
    f.close()
    return x, y,val_x,val_y


# dimensions of our images.
img_width,  img_height =  224,224

epochs = 15
batch_size = 8

model = vgg161_model(img_width,img_height)
#model = get_test_model(img_width,img_height)

exit()
model_checkpoint = ModelCheckpoint('checkpoints/vgg161_224x224_weight.{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_weights_only=True,period=1,save_best_only=False)

train_continue = 1
if train_continue:
    model.load_weights('checkpoints/vgg161_224x224_weight.0.185.hdf5',by_name=False)

mode_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001)

images,masks,val_x,val_y = load()
#images,masks = load()

#model.fit(images,[masks,edge],batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_x,[val_y,edge_val]),callbacks=[model_checkpoint,mode_lr])
model.fit(images,masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_x,val_y),callbacks=[model_checkpoint,mode_lr])

