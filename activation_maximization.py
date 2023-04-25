#!/usr/bin/env python3
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu
from tqdm import tqdm

cnn = load_model('./data/cnns/modis_bt/res_net_encoder_epoch1000')

def multi_scale_noise(N,im=None):
    if im is None:
        im = np.random.uniform(0,1,size=(2,2))
    if im.shape[0]<N:
        sz = im.shape
        im = Image.fromarray(im)
        im = im.resize((sz[0]*2,sz[1]*2),resample=Image.BICUBIC)
        im = np.array(im)
        im += np.random.uniform(0,1,size=im.shape)
        return multi_scale_noise(N,im)
    else:
        im = 80*im/np.max(im)
        return im

def loss_func(image, idx):
    embedding = tf.cast(cnn(image),tf.float32)
    im_loss = -tf.reduce_mean(embedding[0,idx])
    im = image/255.0
    tv_loss =  tf.reduce_mean((im[:,:-1,:,:]-im[:,1:,:,:])**2)
    tv_loss += tf.math.reduce_mean((im[:,:,:-1,:]-im[:,:,1:,:])**2)
    bnd_loss = tf.reduce_mean(relu(image-245)**2 + relu(10-image)**2)
    return im_loss - 100*tv_loss - bnd_loss

@tf.function
def update(img, idx, lr):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = loss_func(img, idx)
    grad = tf.math.l2_normalize(tape.gradient(loss, img))
    img -= lr * grad
    return img

def jitter(img,shift=None):
    img = np.array(img).squeeze()
    if shift is None:
        shift = np.random.randint(-2,3,size=(2,))
    img = np.roll(img, shift[0], axis=0)
    img = np.roll(img, shift[1], axis=1)
    img = tf.convert_to_tensor(img)[np.newaxis,:,:,np.newaxis]
    return img, -shift

def latent_viz(idx):
    img = tf.convert_to_tensor(multi_scale_noise(256)[np.newaxis,:,:,np.newaxis])
    lr = 300
    for iteration in tqdm(range(2500)):
        img, rjit = jitter(img)
        img = update(img, idx, lr)
        img, _ = jitter(img,rjit)
        if iteration==1000:
            lr = 150
        if iteration==2000:
            lr = 50
    img = np.array(img).squeeze()
    img = np.uint8(np.clip(img,0,255))
    return img

for i in range(1024):
    print('Visualizing embedding dim ' + str(i) + '...')
    img = latent_viz(i)
    img = Image.fromarray(img)
    img.save('./data/visualizations/' + str(i).zfill(4) + '.png')
    