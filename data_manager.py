#!/usr/bin/env python3
#
#Andrew V. Geiss, Jun 10th 2022, ICLASS

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from glob import glob
from multiprocess import Pool

#set constants:
SIZE = 'small'
if SIZE =='big':
    CHIP_SIZE_RANGE = [128,512]
    CHIP_SIZE = 256
elif SIZE=='small':
    CHIP_SIZE_RANGE = [96,160]
    CHIP_SIZE = 128

NWORKERS = 12
training_files = glob('./data/goes/training_images/*.npy')
training_files.sort()

def chip_from_image(image):
    
    #select a random chip
    wh = image.shape[1]
    dx, dy = np.random.randint(*CHIP_SIZE_RANGE,size=(2,))
    x0 = np.random.randint(0,wh-dx)
    y0 = np.random.randint(0,wh-dy)
    chip = image[x0:x0+dx,y0:y0+dy]
    
    #random rotations:
    rot = np.random.randint(4)
    chip = np.rot90(chip,k=rot)
    
    #resize the chip
    chip = Image.fromarray(chip)
    chip = chip.resize((CHIP_SIZE,CHIP_SIZE),resample=Image.BICUBIC)
    
    #random flips:
    if np.random.choice([True,False]):
        chip = ImageOps.flip(chip)
    if np.random.choice([True,False]):
        chip = ImageOps.mirror(chip)
    
    #alter the contrast:
    factor = np.random.uniform(0.5,1.5)
    chip = ImageEnhance.Contrast(chip).enhance(factor)
    
    #alter the brightness:
    factor = np.random.uniform(0.5,1.5)
    chip = ImageEnhance.Brightness(chip).enhance(factor)
    
    #gaussian blur:
    if np.random.uniform()<0.15:
        chip = chip.filter(ImageFilter.GaussianBlur(radius=1))
    
    return np.array(chip)

def load_random_image():
    f_ind = np.random.randint(len(training_files))
    image = np.load(training_files[f_ind])
        
    #if we're using small chips, do a random crop:
    if SIZE == 'small':
        wh = image.shape[1]
        d = CHIP_SIZE*4
        x0 = np.random.randint(0,wh-d)
        y0 = np.random.randint(0,wh-d)
        image = image[x0:x0+d,y0:y0+d]
        
    return image

def barlow_batch(N):
    set1, set2 = [],[]
    for i in range(N):
        image = load_random_image()
        set1.append(chip_from_image(image))
        set2.append(chip_from_image(image))
    return np.array(set1)[...,np.newaxis],np.array(set2)[...,np.newaxis]

def chip_inds(IS,CS):
    assert CS<IS
    if IS % CS == 0:
        return list(np.arange(0,IS,CS))
    inds = np.arange(0,IS,CS)
    last = inds[-1]
    inds = inds[:-1]
    if (IS-last)/CS > 0.75:
        inds = np.append(inds,IS-CS)
    return list(inds)

#these functions break up the modis image files into more managable images
def break_up_granule(f):
    print(f,flush=True)
    image = np.array(Image.open(f))
    image_name = f.split('/')[-1]
    chip_count = 0
    for x in [0,1024,2048,3008]:
        try:
            sample = image[x:x+1024,448:1472]
            np.save('./data/modis/training_images/' + image_name + '.' + str(chip_count) + '.npy',sample)
            chip_count += 1
        except:
            print('Problem subsampling image: ' + f)
    
def break_up_granules():
    print('Subsampling Granule Files')
    granule_files = glob('./data/modis/images/2020/*.png')
    granule_files.sort()
    p = Pool(NWORKERS)
    p.map(break_up_granule,granule_files,chunksize=1)
    p.close()

def random_chip_from_image_file(fname):
    print(fname,flush=True)
    image = np.load(fname)
    x,y = np.random.randint(0,image.shape[0]-CHIP_SIZE,size=(2,))
    return image[x:x+CHIP_SIZE,y:y+CHIP_SIZE]

def make_sample_chip_dataset():
    p = Pool(NWORKERS)
    chips = p.map(random_chip_from_image_file,training_files,chunksize=1)
    p.close()
    np.save('./data/modis/chip_dataset.npy',chips)