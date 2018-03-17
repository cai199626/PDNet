import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras
from keras import backend as K
import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import *

def load():
    
    f = h5py.File('data_merge_224x224x2_T.h5','r')    
    f.keys()
    x = f['x'][:]
    y = f['y'][:]
    val_x = f['val_x'][:]
    val_y = f['val_y'][:]
    f.close()
    return x, y,val_x,val_y


# dimensions of our images.
img_width,  img_height =  224,224

nb_validation_samples = 5000
epochs = 15
batch_size = 2

model = vgg161_model(img_width,img_height)

model_checkpoint = ModelCheckpoint('checkpoints/vgg_weight.{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_weights_only=True,period=1,save_best_only=False)

mode_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
images,masks,val_x,val_y = load()

model.fit(images,masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(val_x,val_y),callbacks=[model_checkpoint,mode_lr])

