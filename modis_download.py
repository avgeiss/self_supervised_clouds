#!/usr/bin/env python3
#
#Andrew V. Geiss, Jun 6th 2022, ICLASS
#This script downloads the MODIS training dataset

import multiprocess
from multiprocess import Pool
from pyhdf.SD import SD, SDC
import numpy as np
from urllib.request import urlopen, Request
from glob import glob
import shutil
from time import sleep, time
import sys
from os.path import exists
from PIL import Image

f = open('./modis_auth_token.txt','r')
AUTH_TOKEN = f.readline()
f.close()
GEOLOC_BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/Science%20Domain/Level-1/MODIS%20Level-1/MODIS%20Terra%20C6.1%20-%20Geolocation%20-%201km/'
L1_BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/Science%20Domain/Level-1/MODIS%20Level-1/MODIS%20Terra%20C6.1%20-%20Level%201B%20Calibrated%20Radiances%20-%20500m/'
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')
DATA_DIR = './data/modis/'
TEMP_FILE_DIR = './data/modis/temp/'
NWORKERS = 2
headers = { 'user-agent' : USERAGENT }
headers['Authorization'] = 'Bearer ' + AUTH_TOKEN
import ssl
CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

def trim_viewing_angle_1k(data):
    
    #trims the veiwing angle of 1km data to be less than 45 degrees only
    data = data[:,38*5:-38*5,...]
    
    #trims off a couple extra pixels to get dims divisible by 16:
    data = data[:2016,7:-7,...]
    
    return data

def downsample(data,f):
    sz = data.shape
    data = np.reshape(data,[f,sz[0]//f,f,sz[1]//f],order='F')
    data = np.mean(data,axis=(0,2),keepdims=False)
    return data
    
def extract_granule_metadata(file):
    
    #get the fraction of the granule over ocean > 50m deep:
    ocean_flag = file.select('Land/SeaMask').get()
    ocean_mask = np.logical_or(ocean_flag == 7, ocean_flag==6)
    ocean_mask = downsample(trim_viewing_angle_1k(ocean_mask),16)
    
    #solar zenith
    sza = file.select('SolarZenith').get()/100
    sza = downsample(trim_viewing_angle_1k(sza),16)
    
    #latitude and longitude:
    lat = file.select('Latitude').get()
    lon = file.select('Longitude').get()
    lat = downsample(trim_viewing_angle_1k(lat),16)
    lon = downsample(trim_viewing_angle_1k(lon),16)
    
    return ocean_mask, sza, lat, lon

def extract_500m_550nm(file):
    
    #get the 550nm band:
    data = np.float32(file.select('EV_500_RefSB').get()[1,:,:])
    attr = file.select('EV_500_RefSB').attributes()
    scale = np.float32(attr['reflectance_scales'][0])
    offset = np.float32(attr['reflectance_offsets'][0])
    fill = attr['_FillValue']
    bad_data = data==fill
    data = scale*(data - offset)
    data[bad_data] = 0
    image = np.clip(data*255,0,255)
    image = np.uint8(image)
    
    #trim to viewing angles less than 45 degrees
    crop = (image.shape[1]-1920)//2
    image = image[:4032,crop:-crop]
    
    return image

def download_metadata_file(fname,overwrite=False):
    #get year day and generate file names:
    pid = multiprocess.current_process().pid
    strdate = fname.split('.')[-5]
    year, day = int(strdate[1:5]), int(strdate[5:8])
    file_url = GEOLOC_BASE_URL + str(year) + '/' + str(day).zfill(3) + '/' + fname
    file_path = DATA_DIR + 'metadata/' + str(year) + '/' + fname[:-4] + '.npz'
    
    #check to see if the data was already downloaded:
    if (not overwrite) and exists(file_path):
        return
    
    #download the data
    try:
        fh = urlopen(Request(file_url,headers=headers),context=CTX)
        write_file = open(TEMP_FILE_DIR + 'temp' + str(pid) + '.hdf','w+b')
        shutil.copyfileobj(fh, write_file)
        write_file.close()
    except Exception as e:
        sleep(.5)
        print('Downloading granule metadata ' + fname + '... Download failed!',flush=True)
        print(e,flush=True)
        return
    
    #convert required fields to numpy format:
    try:
        hdf_file = SD(TEMP_FILE_DIR + 'temp' + str(pid) + '.hdf', SDC.READ)
        metadata = np.array(extract_granule_metadata(hdf_file))
        np.savez_compressed(file_path,np.array(metadata,dtype='float16'))
    except Exception as e:
        sleep(.5)
        print('Downloading granule metadata ' + fname + '... File conversion failed',flush=True)
        print(e,flush=True)
        return
    
    print('Downloading granule metadata ' + fname + '... Success',flush=True)
    
def download_image_file(fname,overwrite=False):
    #get year day and generate file names:
    pid = multiprocess.current_process().pid
    strdate = fname.split('.')[-5]
    year, day = int(strdate[1:5]), int(strdate[5:8])
    file_url = L1_BASE_URL + str(year) + '/' + str(day).zfill(3) + '/' + fname
    file_path = DATA_DIR + 'images/' + str(year) + '/' + fname[:-4] + '.png'
    
    #check to see if the data was already downloaded:
    if (not overwrite) and exists(file_path):
        return
    #download the data
    try:
        t1 = time()
        fh = urlopen(Request(file_url,headers=headers),context=CTX)
        write_file = open(TEMP_FILE_DIR + 'temp' + str(pid) + '.hdf','w+b')
        shutil.copyfileobj(fh, write_file)
        write_file.close()
        elapsed = np.round(time()-t1,2)
    except Exception as e:
        sleep(.5)
        print('Downloading granule data ' + fname + '... Download failed!',flush=True)
        print(e,flush=True)
        return
    
    #convert required fields to numpy format:
    try:
        hdf_file = SD(TEMP_FILE_DIR + 'temp' + str(pid) + '.hdf', SDC.READ)
        image_data = np.array(extract_500m_550nm(hdf_file))
        image = Image.fromarray(image_data)
        image.save(file_path)
    except Exception as e:
        sleep(.5)
        print('Downloading granule data ' + fname + '... File conversion failed, Download Time: ' + str(elapsed),flush=True)
        print(e,flush=True)
        return
    
    print('Downloading granule metadata ' + fname + '... Success, Download Time: ' + str(elapsed),flush=True)

def download_metadata():
    list_file = open(DATA_DIR + 'metadata_file_list.txt','r')
    file_names = list_file.readlines()
    list_file.close()
    file_names = [f[:-1] for f in file_names]
    p = Pool(NWORKERS)
    p.map(download_metadata_file,file_names,chunksize=1)
    p.close()
    
def download_images():
    list_file = open(DATA_DIR + 'download_file_list.txt','r')
    file_names = list_file.readlines()
    list_file.close()
    file_names = [f[:-1] for f in file_names]
    p = Pool(NWORKERS)
    p.map(download_image_file,file_names,chunksize=1)
    p.close()
            
def get_metadata_file_list(year):
    list_file = open(DATA_DIR + 'metadata_file_list.txt','w')
    for day in range(1,367):
        print('Downloading file list for ' + str(year) + ' day ' + str(day))
        response = urlopen(GEOLOC_BASE_URL + str(year) + '/' + str(day).zfill(3) + '.csv')
        file_names = [line.decode('utf-8').rstrip().split(',')[0]+'\n' for line in response]
        file_names.pop(0)
        file_names.sort()
        list_file.writelines(file_names)
    list_file.close()
    
def get_l1_file_list(year):
    list_file = open(DATA_DIR + 'l1_file_list.txt','w')
    for day in range(1,367):
        print('Downloading file list for ' + str(year) + ' day ' + str(day))
        response = urlopen(L1_BASE_URL + str(year) + '/' + str(day).zfill(3) + '.csv')
        file_names = [line.decode('utf-8').rstrip().split(',')[0]+'\n' for line in response]
        file_names.pop(0)
        file_names.sort()
        list_file.writelines(file_names)
    list_file.close()
    
def select_valid_granules(year):
    #checks to see if we have metadata downloaded for each image available on the server
    #checks that the file meets SZA and ocean fraction criteria and adds to a list of files to download
    
    def tstamp(fname):
        #extracts the time stamp from a MODIS L1 file name
        split = fname.split('.')
        return split[-5][1:] + split[-4]
    
    metadata_files = [f.split('/')[-1] for f in glob(DATA_DIR + 'metadata/' + str(year) + '/*.npz')]
    metadata_files.sort()
    metadata_tstamps = [tstamp(f) for f in metadata_files]
    l1_file_list = open(DATA_DIR + 'l1_file_list.txt','r')
    l1_files = [l[:-1] for l in l1_file_list.readlines()]
    l1_files.sort()
    
    list_file = open(DATA_DIR + 'download_file_list.txt','w')
    print('Getting usable granule list:')
    for f in l1_files:
        l1_tstamp = tstamp(f)
        if not (l1_tstamp in metadata_tstamps):
            print('No data for granule: ' + f)
        else:
            fidx = metadata_tstamps.index(l1_tstamp)
            fname = metadata_files[fidx]
            metadata = np.load(DATA_DIR + 'metadata/' + str(year) + '/' + fname)['arr_0']
            metadata = np.mean(metadata[:2,:,:],axis=(1,2),keepdims=False)
            ocean_frac, sza = metadata[0],metadata[1]
            if ocean_frac>0.9 and sza<60:
                list_file.write(f + '\n')
    list_file.close()
    print('...done')
        
YEAR = 2020
get_metadata_file_list(YEAR)
download_metadata()
get_l1_file_list(YEAR)
select_valid_granules(YEAR)
download_images()