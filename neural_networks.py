#!/usr/bin/env python3
#
#Andrew V. Geiss, Jun 10th 2022, ICLASS

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, Conv2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow import repeat
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.applications import ResNet50, DenseNet121
import tensorflow as tf
import numpy as np

LAMBDA = 0.005

def barlow_loss(_,C):
    N = C.shape[0]
    E = tf.eye(N)
    CE = (E-C)*(E-C)
    diag = tf.math.reduce_sum(CE*E)
    off_diag = tf.math.reduce_sum(CE*(1-E))
    return diag + LAMBDA*off_diag

def cross_correlation(x,y,batch_size):
    
    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    
    def normalize(z):
        z = z-tf.math.reduce_mean(z,axis=0,keepdims=True)
        z = z/tf.math.reduce_std(z,axis=0,keepdims=True)
        return z
    
    #normalize along batch dimension:
    x = normalize(x)
    y = normalize(y)
    
    #compute cross correlation matrix:
    C = tf.linalg.matmul(tf.transpose(x),y)/batch_size
    
    return C
        
def barlow_model(encoder,projector,batch_size):
    
    #make the barlow model:
    insz = tuple(np.array(encoder.layers[0].input.shape)[1:])
    chip1 = Input(insz)
    chip2 = Input(insz)
    proj1 = projector(encoder(chip1))
    proj2 = projector(encoder(chip2))
    C = cross_correlation(proj1,proj2,batch_size)
    
    cnn = Model([chip1,chip2],C)
    cnn.compile(optimizer=LAMB(learning_rate=0.0005),loss=barlow_loss)
    
    return cnn

def MLP(input_size, layer_sizes):
    xin = Input((input_size,))
    x = Dense(layer_sizes[0])(xin)
    for l in layer_sizes[1:]:
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(l)(x)
    return Model(xin,x)
    
def res_layer(x, in_chan, out_chan, downsample=False):
    
    def br(x):
        return ReLU()(BatchNormalization()(x))
    
    def conv(x,n,c):
        return Conv2D(c,(n,n),padding='same')(x)
    
    if downsample:
        x = br(AveragePooling2D((2,2))(x))
        xpass = conv(x,1,out_chan)
        x = conv(x,1,in_chan)
    else:
        xpass = x
        x = conv(br(x),1,in_chan)
    
    x = conv(br(x),3,in_chan)
    x = conv(br(x),1,out_chan)
        
    return x + xpass

def res_block(x,internal_chan, output_chan, n_layers):
    x = res_layer(x,internal_chan, output_chan, downsample=True)
    for _ in range(n_layers-1):
        x = res_layer(x,internal_chan, output_chan)
    return x

def res_net():
    xin = Input((256,256,1))
    xin = xin/127.5-1
    x = Conv2D(64,(7,7),strides=(2,2),padding='same')(xin)
    x = res_block(x, 64, 256, 3)
    x = res_block(x, 128, 256, 3)
    x = res_block(x, 256, 512, 3)
    x = res_block(x, 256, 1024, 3)
    x = res_block(x, 512, 1024, 3)
    x = GlobalAveragePooling2D()(x)
    return Model(xin,x)

def pretrained(input_width,cnn_name):
    if cnn_name == 'RN50':
        cnn = ResNet50(include_top=False,input_shape=(input_width,input_width,3))
    elif cnn_name == 'DN121':
        cnn = DenseNet121(include_top=False,input_shape=(input_width,input_width,3))
    xin = Input((input_width,input_width,1))
    x = repeat(xin,3,axis=-1)
    x = cnn(x)
    x = GlobalAveragePooling2D()(x)
    return Model(xin,x)