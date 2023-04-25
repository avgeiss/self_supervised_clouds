#!/usr/bin/env python3
#utils.py

from sklearn.metrics import confusion_matrix
import numpy as np
from numba import njit, prange

def str_prc(x):
    x = np.round(x*100,1)
    return str(x) + '%'

#prints a nicely formatted confusion matrix:
def confusion(y_true,y_pred,names):
    cat_acc = np.mean(y_true==y_pred)
    cm = confusion_matrix(y_true,y_pred,normalize='true')
    nspc = np.max([len(name) for name in names])
    print(' '*nspc,end=' | ')
    for i in range(len(names)):
        print(names[i].rjust(nspc),end=' | ')
    print('')
    for i in range(len(names)):
        print(names[i].ljust(nspc),end=' | ')
        for j in range(len(names)):
            if i == j:
                print(str_prc(cm[i,j]).rjust(nspc),end=' | ')
            else:
                print(str_prc(cm[i,j]).rjust(nspc),end=' | ')
            
        print('')
    print('\nCategorical Accuracy: ' + str_prc(cat_acc))
    return cat_acc, cm

def cos_dist(x,y):
    return 1.001-np.dot(x,y.T)

def to_unit_vecs(X):
    return X/(np.sum(X**2,axis=-1,keepdims=True)**0.5)

@njit(parallel=True)
def hist2d(x,y,xbins,ybins):
    N = len(x)
    NB = len(xbins)
    counts = np.zeros((NB-1,NB-1),dtype='double')
    for i in prange(N):
        dx, dy = x[i], y[i]
        if np.isnan(dx) or np.isnan(dy):
            continue
        ix, iy = 1, 1
        while xbins[ix] < dx:
            ix += 1
        while ybins[iy] < dy:
            iy += 1
        counts[ix-1,iy-1] += 1
    return counts

def TC_to_reflectance(im):
    im = np.double(im)
    im[im<=110] = 30*im[im<=110]/110
    im[(im>110)*(im<=160)] = (im[(im>110)*(im<=160)]-110)*(30/50)+30
    im[(im>160)*(im<=210)] = (im[(im>160)*(im<=210)]-160)*(60/50)+60
    im[(im>210)*(im<=240)] = (im[(im>210)*(im<=240)]-210)*(70/30)+120
    im[(im>240)*(im<=255)] = (im[(im>240)*(im<=255)]-240)*(65/15)+190
    im = np.uint8(im)
    return im

def reflectance_to_TC(im):
    im = np.double(im)
    im[im>190] = (im[im>190]-190)*(15/65) + 240
    im[(im>120)*(im<=190)] = (im[(im>120)*(im<=190)] - 120)*(30/70) + 210
    im[(im>60)*(im<=120)] = (im[(im>60)*(im<=120)]-60)*(50/60)+160
    im[(im>30)*(im<=60)] = (im[(im>30)*(im<=60)]-30)*(50/30)+110
    im[im<=30] = im[im<=30]*110/30
    im = np.uint8(im)
    return im
