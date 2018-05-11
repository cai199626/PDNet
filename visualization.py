import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from model import *
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def load():
    #f = h5py.File('data_lfsd_224x224.h5','r')    
    #f = h5py.File('test_nlpr_224x224.h5','r')    
    f = h5py.File('test_nju500_224x224.h5','r')    
    f.keys()
    X = f['x'][:]
    #y = f['y'][:]
    f.close()
    return X



images = load()
image = images[:,:,:,0:3]
print image.shape
deep = images[:,:,:,3:4]
print deep.shape


# dimensions of our images.
img_width,  img_height = 224,224 
#mask_width, mask_height = 120, 120


################################################################################
#TN=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

#model = get_model(img_width,img_height)
model = vgg16_deep_fuse_model(img_width,img_height)
#model = vgg161_model(img_width,img_height)

model.load_weights('checkpoints/vgg16_deep_fuse_768.0.130.hdf5',by_name=False)
#model.load_weights('checkpoints/msra_96x96_weight.0.19.hdf5',by_name=False)
#model.load_weights('checkpoints/new_fine_msra_96x96_weight.0.165.hdf5',by_name=False)
#model.load_weights('checkpoints/new2_vgg_msra_192x192x2_weight.0.184.hdf5',by_name=False)
layer_of_interest=1
thisInput = image[0]
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediate_tensor = intermediate_tensor_function([thisInput])
#intermediate_tensor = intermediate_tensor_function([thisInput])[0]

print intermediate_tensor.shape
#img_pre=model.predict([image,deep],batch_size=2, verbose=1)
#img_pre=model.predict([image,deep],batch_size=8, verbose=1)
#for i in range(img_pre.shape[0]):
#    #if i>200:
#        #break
#    img = img_pre[i]
#    img = array_to_img(img)
#    img.save("results/%04d.png"%(1+i))

