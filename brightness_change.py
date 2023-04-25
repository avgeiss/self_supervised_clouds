#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from utils import TC_to_reflectance, to_unit_vecs, cos_dist
from tensorflow.keras.models import load_model

#set the brightness scaling factors
factors = np.array([50,55,60,65,70,75,80,85,90,93,95,97,98,99,100,101,102,103,105,107,110,115,120,125,130,135,140,145,150])/100

def load_image(fname):
    im = Image.open(fname)
    im = np.array(im)
    if im.ndim == 3:
        im = TC_to_reflectance(im[:,:,1])
    return im

#converts a matrix of embeddings to unit vectors
def unit_vec(x):
    return x/np.sqrt(np.sum(x**2,axis=-1,keepdims=True))

def load_ims_by_idx(fnames, inds):
    images = [load_image(fnames[i]) for i in tqdm(inds)]
    return np.stack(images)

def calc_brightness_adj(dataset):
    if dataset == 'yuan2020':
        chip_fnames = sorted(glob('./data/yuan2020/chips/*/*.png'))
    elif dataset == 'rasp2020':
        chip_fnames = sorted(glob('./data/rasp2020/chips/*/*.png'))
    #load the dataset
    fnames = np.array(chip_fnames)
    images = load_ims_by_idx(fnames,np.arange(len(fnames)))
    cnn = load_model('./data/cnns/modis_bt/res_net_encoder_epoch1000')
    emb = cnn.predict(images,batch_size=128)
    emb = np.double(emb)
    emb = to_unit_vecs(emb)
    dists = cos_dist(emb, emb)
    nn_dist = 1.001-np.sort(dists,axis=0)[1,:]
    self_dists = []
    for f in factors:
        adj_ims = np.uint8(np.clip(np.double(images)*f,0,255))
        adj_emb = cnn.predict(adj_ims,batch_size=512)
        adj_emb = to_unit_vecs(np.double(adj_emb))
        self_dists.append(np.sum(adj_emb*emb,axis=1))
    return self_dists, nn_dist

plt.figure(figsize=(14,6.5))
plt.subplot(1,2,1)
perc = factors*100-100
self_dists, nn_dist = calc_brightness_adj('rasp2020')
self_dists = np.array(self_dists)
prctile = np.percentile(self_dists,[5,95],axis=1)
plt.fill_between(perc,prctile[0],prctile[1],color='b',alpha=0.25)
prctile = np.percentile(self_dists,[25,75],axis=1)
plt.fill_between(perc,prctile[0],prctile[1],color='b',alpha=0.25)
plt.plot(perc,np.percentile(self_dists,50,axis=1),color='b')
prctile = np.percentile(nn_dist,[5,95])
plt.fill_between(perc[[0,-1]],prctile[0]*np.array([1,1]),prctile[1]*np.array([1,1]),color='r',alpha=0.25)
prctile = np.percentile(nn_dist,[25,75])
plt.fill_between(perc[[0,-1]],prctile[0]*np.array([1,1]),prctile[1]*np.array([1,1]),color='r',alpha=0.25)
plt.plot(perc[[0,-1]],np.mean(nn_dist)*np.array([1,1]),'r-')
plt.xlim([-50,50])
plt.ylim([1-0.15,1])
plt.xlabel('Brightness Adjustment (%)')
plt.ylabel('Cosine Similarity')
plt.title('SGFF Dataset')

plt.subplot(1,2,2)
self_dists, nn_dist = calc_brightness_adj('yuan2020')
self_dists = np.array(self_dists)
prctile = np.percentile(self_dists,[5,95],axis=1)
plt.fill_between(perc,prctile[0],prctile[1],color='b',alpha=0.25)
prctile = np.percentile(self_dists,[25,75],axis=1)
plt.fill_between(perc,prctile[0],prctile[1],color='b',alpha=0.25)
plt.plot(perc,np.percentile(self_dists,50,axis=1),color='b')
prctile = np.percentile(nn_dist,[5,95])
plt.fill_between(perc[[0,-1]],prctile[0]*np.array([1,1]),prctile[1]*np.array([1,1]),color='r',alpha=0.25)
prctile = np.percentile(nn_dist,[25,75])
plt.fill_between(perc[[0,-1]],prctile[0]*np.array([1,1]),prctile[1]*np.array([1,1]),color='r',alpha=0.25)
plt.plot(perc[[0,-1]],np.mean(nn_dist)*np.array([1,1]),'r-')
plt.xlim([-50,50])
plt.ylim([1-0.15,1])
plt.xlabel('Brightness Adjustment (%)')
plt.ylabel('Cosine Similarity')
plt.title('Yuan et al. 2020 Dataset')
plt.savefig('./figures/brightness_adj.png',dpi=500)