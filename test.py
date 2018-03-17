import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator,array_to_img
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from model import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def load():
    f = h5py.File('data_ecssd_224x224x2_T.h5','r')    
    #f = h5py.File('../data_omron_224x224x2_T.h5','r')    
    #f = h5py.File('data_sod_224x224x2_T.h5','r')    
    #f = h5py.File('../data_pascal_224x224x2_T.h5','r')    
    #f = h5py.File('../data_duts_te_224x224x2_T.h5','r')    
    #f = h5py.File('../data_huk_224x224x2_T.h5','r')    
    #f = h5py.File('../data_test_224x224x2_T.h5','r')    
 
    #f = file('data_salcon_T.cPickle', 'rb')
    #loaded_obj = pickle.load(f)
    #f.close()
    #X, y = loaded_obj
    f.keys()
    X = f['x'][:]
    y = f['y'][:]
    f.close()
    return X, y



images,masks = load()

# dimensions of our images.
img_width,  img_height = 224,224 
#mask_width, mask_height = 120, 120


nb_train_samples = 1000
nb_validation_samples = 0
epochs = 120
batch_size = 128

################################################################################
#TN=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

#model = get_model(img_width,img_height)
model = vgg161_model(img_width,img_height)

model.load_weights('vgg161_merge_224x224x2_weight.0.168.hdf5',by_name=False)

img_pre=model.predict(images,batch_size=10, verbose=1)
for i in range(img_pre.shape[0]):
    #if i>200:
        #break
    img = img_pre[i]
    img = array_to_img(img)
    img.save("results/%04d.png"%(1+i))

