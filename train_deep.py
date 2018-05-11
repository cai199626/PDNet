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

model = vgg16_deep_fuse_model(img_width,img_height)
#model = vgg16_deep_model(img_width,img_height)
#model = get_test_model(img_width,img_height)

#exit()
model_checkpoint = ModelCheckpoint('checkpoints/vgg16_deep_fuse_512_no_prior.{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_weights_only=True,period=1,save_best_only=False)

train_continue = 0 
if train_continue:
    model.load_weights('checkpoints/vgg161_nju_224x224_weight.0.217.hdf5',by_name=False)

mode_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001)

#images,masks,val_x,val_y = load()
images,masks ,val_images,val_y= load()


val_image = val_images[:,:,:,0:3]
print val_image.shape
val_deep = val_images[:,:,:,3:4]
print val_deep.shape


image = images[:,:,:,0:3]
print image.shape
deep = images[:,:,:,3:4]
print deep.shape

model.fit([image,deep],masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=([val_image,val_deep],val_y),callbacks=[model_checkpoint,mode_lr])
#model.fit([image,deep],masks,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2,callbacks=[model_checkpoint,mode_lr])

