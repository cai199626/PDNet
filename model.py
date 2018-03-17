import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import h5py

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import cPickle as pickle

import itertools


def bce_loss(y_true, y_pred):
    return tf.sqrt( tf.reduce_sum(tf.square(y_pred - y_true ) ) )


def vgg161_model(img_width,img_height):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_height, img_width)
    else:
        input_shape = (img_height, img_width,3)
    print K.image_data_format()

    inputs=Input(input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='TruncatedNormal',name='block1_conv1')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)
    
    maxpool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='TruncatedNormal',name='block2_conv1')(maxpool1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    maxpool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), padding='same', kernel_initializer='TruncatedNormal', name='block3_conv1')(maxpool2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    conv3_4 = Conv2D(256, (3, 3), padding='same', name='block3_conv4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    conv3_4 = Activation('relu')(conv3_4)
    maxpool3 = MaxPooling2D((2, 2), name='block3_pool')(conv3_4)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), padding='same', kernel_initializer='TruncatedNormal', name='block4_conv1')(maxpool3)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_3 = Activation('relu')(conv4_3)
    conv4_4 = Conv2D(512, (3, 3), padding='same', name='block4_conv4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    conv4_4 = Activation('relu')(conv4_4)
    maxpool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),   padding='same', kernel_initializer='TruncatedNormal',name='block5_conv1')(maxpool4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)
    conv5_2 = Conv2D(512, (3, 3),   padding='same', name='block5_conv2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Activation('relu')(conv5_2)
    conv5_3 = Conv2D(512, (3, 3),   padding='same', name='block5_conv3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_3 = Activation('relu')(conv5_3)

    # maxpool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_3)
    # conv5_3 = Conv2D(1024, (3, 3),   padding='same', name='block6_conv3')(maxpool5)
    # conv5_3 = BatchNormalization()(conv5_3)
    # conv5_3 = Activation('relu')(conv5_3)

    # conv5_3 = Conv2D(1024, (3, 3),   padding='same', name='block6_conv')(conv5_3)
    # conv5_3 = BatchNormalization()(conv5_3)
    # conv5_3 = Activation('relu')(conv5_3)
    # uppool0=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv5_3)

    #de conv begin
    conv6_1 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_1',kernel_initializer='TruncatedNormal',)(conv5_3)
    conv6_1 = BatchNormalization()(conv6_1)
    conv6_1 = Activation('relu')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_2',kernel_initializer='TruncatedNormal',)(conv6_1)
    conv6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Activation('relu')(conv6_2)
    conv6_3 = Conv2D(512, (3, 3),   padding='same', name='de_conv5_3',kernel_initializer='TruncatedNormal',)(conv6_2)
    conv6_3 = BatchNormalization()(conv6_3)
    conv6_3 = Activation('relu')(conv6_3)

    uppool1=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv6_3)

    #cross 1
    merge1 = concatenate([conv4_4,uppool1],axis=-1)
    conv_m1 = Conv2D(512, (3, 3),   padding='same', kernel_initializer='TruncatedNormal')(merge1)
    conv_m1 = BatchNormalization()(conv_m1)
    conv_m1 = Activation('relu')(conv_m1)
    conv7_1 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_1',kernel_initializer='TruncatedNormal',)(conv_m1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = Activation('relu')(conv7_1)
    conv7_2 = Conv2D(512, (3, 3),   padding='same', name='de_conv4_2',kernel_initializer='TruncatedNormal',)(conv7_1)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = Activation('relu')(conv7_2)
    
    uppool2=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv7_2)
    uppool2_1=UpSampling2D(size=(4, 4), data_format=K.image_data_format())(uppool2)
    outputs2_1=Conv2D(1, (3, 3),padding='same',kernel_initializer='TruncatedNormal')(uppool2_1)

    #cross 2
    merge2 = concatenate([conv3_4,uppool2],axis=-1)
    conv_m2 = Conv2D(256, (3, 3),   padding='same', kernel_initializer='TruncatedNormal')(merge2)
    conv_m2 = BatchNormalization()(conv_m2)
    conv_m2 = Activation('relu')(conv_m2)
    conv8_1 = Conv2D(256, (3, 3),   padding='same', name='de_conv3_1',kernel_initializer='TruncatedNormal',)(conv_m2)
    conv8_1 = BatchNormalization()(conv8_1)
    conv8_1 = Activation('relu')(conv8_1)
    
    uppool3=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv8_1)  
    uppool3_1=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(uppool3)
    outputs3_1=Conv2D(1, (3, 3),padding='same',kernel_initializer='TruncatedNormal')(uppool3_1)
    
    #cross 3
    merge3 = concatenate([conv2_2,uppool3],axis=-1)
    conv_m3 = Conv2D(128, (3, 3),   padding='same', kernel_initializer='TruncatedNormal')(merge3)
    conv_m3 = BatchNormalization()(conv_m3)
    conv_m3 = Activation('relu')(conv_m3)
    conv9_1 = Conv2D(128, (3, 3),   padding='same', name='de_conv2_1',kernel_initializer='TruncatedNormal',)(conv_m3)
    conv9_1 = BatchNormalization()(conv9_1)
    conv9_1 = Activation('relu')(conv9_1)
    
    uppool4=UpSampling2D(size=(2, 2), data_format=K.image_data_format())(conv9_1)
    outputs4_1=Conv2D(1, (3, 3),padding='same',kernel_initializer='TruncatedNormal')(uppool4)
    
    #cross 4
    merge3 = concatenate([conv1_2,uppool4],axis=-1)
    conv_m4 = Conv2D(64, (3, 3),   padding='same', kernel_initializer='TruncatedNormal')(merge3)
    conv_m4 = BatchNormalization()(conv_m4)
    conv_m4 = Activation('relu')(conv_m4)
    # conv10_1 = Conv2D(64, (3, 3),   padding='same', name='de_conv1_1',kernel_initializer='TruncatedNormal',)(uppool4)
    # conv10_1 = BatchNormalization()(conv10_1)
    # conv10_1 = Activation('relu')(conv10_1)
    conv10_2 = Conv2D(64, (3, 3),   padding='same', name='de_conv1_2',kernel_initializer='TruncatedNormal',)(conv_m4)
    conv10_2 = BatchNormalization()(conv10_2)
    conv10_2 = Activation('relu')(conv10_2)
    outputs=Conv2D(1, (3, 3),padding='same',kernel_initializer='TruncatedNormal')(conv10_2)
    
    concat_out=concatenate([outputs,outputs4_1,outputs3_1,outputs2_1])

    outputs_all=Conv2D(1, (3, 3),padding='same',kernel_initializer='TruncatedNormal',activation='sigmoid')(concat_out)
    
    model = Model(inputs = inputs, outputs = outputs_all)

#model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True) 
    #rms=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    sgd=keras.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)#,clipvalue=5)
    adam1=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='binary_crossentropy',#'mean_squared_error',
#loss_weights=[1, 0.4,0.3,0.3],
#optimizer=sgd,
#optimizer='adam',
                  optimizer=adam1)
#model.summary()
    return model

