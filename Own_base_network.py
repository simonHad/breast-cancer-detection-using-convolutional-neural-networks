#!/usr/bin/env python
# coding: utf-8

# ### Own Base line CNN Architecture for feature extraction

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#import os
#os.chdir('/content/drive/My Drive/MSC_BCD_Model')  #change dir
#!ls
#!pwd


# #### Import requeried module  and library

# In[13]:


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


# In[17]:


epochs = 500
INIT_LR = 1e-4
Batch_size = 32
directory_root = 'data'
if K.image_data_format() == "channels_last":
    bn_axis = 3
else:
    bn_axis = 1


# In[5]:


def get_weight_path():
    if K.image_data_format() == "channels_first":
        print('pretrained weights not available for Own base network with theano backend')
        return
    else:
        return 'Own_base_network_model.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)  


# In[6]:


def nn_base(input_tensor=None, trainable=False):

    # Determine proper input shape
    if K.image_data_format() == "channels_first":
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        inputShape = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputShape = Input(tensor=input_tensor, shape=input_shape)
        else:
            inputShape = input_tensor


    # ConvLayer_Block 1
    x= Conv2D(64,(3,3), activation='relu',padding='same', name='block_1_conv1')(inputShape)
    x= BatchNormalization(axis=bn_axis,name='block_1_batchNormal')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block_1_pool')(x)
    x= Dropout(0.25,name='block_1_dropout')(x)
    # ConvLayer_Block 2
    x= Conv2D(128,(3,3), activation='relu',padding='same', name='block_2_conv1')(x)
    x= BatchNormalization(axis=bn_axis,name='block_2_batchNormal')(x)
    # ConvLayer_Block 3
    x= Conv2D(256,(3,3), activation='relu',padding='same', name='block_3_conv1')(x)
    x= BatchNormalization(axis=bn_axis,name='block_3_batchNormal')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block_3_pool')(x)
    x= Dropout(0.25,name='block_3_dropout')(x)
    # ConvLayer_Block 4
    x= Conv2D(512,(3,3), activation='relu',padding='same', name='block_4_conv1')(x)
    x= BatchNormalization(axis=bn_axis,name='block_4_batchNormal')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block_4_pool')(x)
    x= Dropout(0.25,name='block_4_dropout')(x)
    # ConvLayer_Block 5
    x= Conv2D(512,(3,3), activation='relu',padding='same', name='block_5_conv1')(x)
    x= BatchNormalization(axis=bn_axis,name='block_5_batchNormal')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block_5_pool')(x)
    x= Dropout(0.25,name='block_5_dropout')(x)

    return x


# In[8]:


def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x= BatchNormalization(axis=bn_axis,name='RPN_batchNormal')(x)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


# In[ ]:


def classifier(base_layers, input_rois, num_rois, nb_classes, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 5
        input_shape = (num_rois,5,5,512)
    elif K.backend() == 'theano':
        pooling_regions = 5
        input_shape = (num_rois,512,5,5)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(1024, activation='relu', name='dense_layer_1'))(out)
    out = TimeDistributed(BatchNormalization(axis=bn_axis,name='batchNormal_1'))(out)
    out = TimeDistributed(Dropout(0.25))(out)
    out = TimeDistributed(Dense(1024, activation='relu', name='dense_layer_2'))(out)
    out = TimeDistributed(BatchNormalization(axis=bn_axis,name='batchNormal_2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

