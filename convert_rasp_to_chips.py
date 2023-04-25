#!/usr/bin/env python3

#convert_rasp2020_to_chips.py
import numpy as np
from glob import glob
from PIL import Image
from multiprocess import Pool
import os
classes = ['Sugar','Flower','Fish','Gravel']
n_keep = 2200 #the number of randomly selected samples to keep from each class (balancing)

#get a list of the modis images:
modis_files = sorted(glob('./data/rasp2020/modis_images/*.png'))

#function to parse a single line in the annotation file:
def parse_anno_line(line):
    _,fname,xstart,ystart,xend,yend,label = line.split(',')
    fname = fname.split('_')[2].split('/')[0] + '_R' + fname[6] + '_' + fname.split('_')[3][-8:]
    if xend == '':
        box = None
    else:
        box = ((int(xstart),int(xend)),(int(ystart),int(yend)))
    label = label[:-1]
    return fname, box, label

#read in the annotation data
f = open('./data/rasp2020/annotations.csv','r')
anno_lines = f.readlines()
f.close()
annotations = [parse_anno_line(line) for line in anno_lines]

#performs an efficient 256x256 boxcar sum to identify contiguous regions with
#enough consistent cloud labels
def box_sum_2d(im):

    def box_sum_1d(im):
        N = 256
        im = np.pad(im,((0,N),(0,0)))
        im = np.flip(np.cumsum(np.flip(im,axis=0),axis=0),axis=0)
        im = im[:-N,:]-im[N:,:]
        return im
    
    im = box_sum_1d(im)
    im = box_sum_1d(np.transpose(im))
    im = np.transpose(im)
    return im

#extracts consistently labeled image chips from each MODIS image
#consistent = 3 labelers agree + no other labels
def proc_file(fname):
    
    #load the MODIS image:
    print(fname)
    image_name = fname.split('/')[-1].split('.')[0]
    image = np.array(Image.open(fname))
    
    #create an 4-channel image mask from the annotations:
    annos = [a for a in annotations if a[0] == image_name and a[1] != None and a[2] != '']
    labels = np.zeros(shape=(*image.shape[:2],len(classes)))
    for a in annos:
        ((x0,x1),(y0,y1)), label = a[1:]
        labels[y0:y1,x0:x1,classes.index(label)] += 1
    
    #ignore labels in missing data regions (between MODIS swaths)
    missing = (image[:,:,0]==0)*(image[:,:,1]==0)*(image[:,:,2]==0)
    for i in range(len(classes)):
        #generate valid mask for current class
        mask = labels[:,:,i]
        mask[missing] = 0
        mask[np.sum(labels,axis=-1) > mask] = 0
        mask = np.double(mask>2)
        #extract chips from contiguous regions where labelers are in agreement
        counts = box_sum_2d(mask)
        while np.any(counts==256**2):
            inds = np.where(counts==256**2)
            y,x = inds[0][0], inds[1][0]
            sample = image[y:y+256,x:x+256,:]
            sample = Image.fromarray(sample)
            sample.save('./data/rasp2020/chips/' + classes[i] + '/' + image_name + '_' + str(y) + '_' + str(x) + '.png')
            mask[y:y+256,x:x+256] = 0
            counts = box_sum_2d(mask)
           
#operating on an SSD so this can be parallelized for faster processing
p = Pool(24)
p.map(proc_file,modis_files,chunksize=1)
p.close()

#delete the excess cases:
np.random.seed(0)
for c in classes:
    fnames = sorted(glob('./data/rasp2020/chips/' + c + '/*.png'))
    fnames = list(np.random.choice(fnames,len(fnames)-n_keep,replace=False))
    for f in fnames:
        os.system('rm ' + f)