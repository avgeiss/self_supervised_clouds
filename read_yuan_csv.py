#retrieves 500m resolution images for the Yuan et al. 2020 dataset

from glob import glob
import numpy as np
from urllib.request import urlopen, Request
from tqdm import tqdm
import sys
L1_BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/Science%20Domain/Level-1/MODIS%20Level-1/MODIS%20Aqua%20C6.1%20-%20Level%201B%20Calibrated%20Radiances%20-%20500m/'
f = open('./modis_auth_token.txt','r')
AUTH_TOKEN = f.readline()
f.close()
GEOLOC_BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/Science%20Domain/Level-1/MODIS%20Level-1/MODIS%20Aqua%20C6.1%20-%20Geolocation%20-%201km/'
L1_BASE_URL = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/Science%20Domain/Level-1/MODIS%20Level-1/MODIS%20Aqua%20C6.1%20-%20Level%201B%20Calibrated%20Radiances%20-%20500m/'
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')
DATA_DIR = './data/modis/yuan_granules/'
TEMP_FILE_DIR = './data/modis/temp/'
NWORKERS = 2
headers = { 'user-agent' : USERAGENT }
headers['Authorization'] = 'Bearer ' + AUTH_TOKEN
import ssl
CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
import shutil
from pyhdf.SD import SD, SDC
from PIL import Image

def get_l1_file_list(year,day):
    response = urlopen(L1_BASE_URL + str(year) + '/' + str(day).zfill(3) + '.csv')
    file_names = [line.decode('utf-8').rstrip().split(',')[0] for line in response]
    file_names.pop(0)
    file_names.sort()
    return file_names

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
    
    return image

def download_image_file(fname,overwrite=False):
    #get year day and generate file names:
    strdate = fname.split('.')[-5]
    year, day = int(strdate[1:5]), int(strdate[5:8])
    file_url = L1_BASE_URL + str(year) + '/' + str(day).zfill(3) + '/' + fname
    fh = urlopen(Request(file_url,headers=headers),context=CTX)
    write_file = open(TEMP_FILE_DIR + 'temp.hdf','w+b')
    shutil.copyfileobj(fh, write_file)
    write_file.close()
    hdf_file = SD(TEMP_FILE_DIR + 'temp.hdf', SDC.READ)
    image_data = np.array(extract_500m_550nm(hdf_file))
    image = Image.fromarray(image_data)
    image.save('./data/yuan2020/hr_granules/' + fname + '.png')


def download_hr_dataset():
    files = glob('./data/yuan2020/chips/*/*.png')
    myd_files = [f.split('/')[-1][10:22] for f in files]
    granules = list(np.unique(myd_files))
    dates = list(np.unique([f[:7] for f in granules]))
    
    hk_fnames = []
    for date in tqdm(dates):
        year = int(date[:4])
        day = int(date[4:])
        hk_fnames += get_l1_file_list(year, day)
    
    from tqdm import tqdm
    for granule in tqdm(granules):
        hk_fname = [f for f in hk_fnames if granule in f][0]
        download_image_file(hk_fname)
    

import matplotlib.pyplot as plt
files = glob('./data/yuan2020/chips_128/*/*.png')
granule_fnames = glob('./data/yuan2020/hr_granules/*.png')
for f in tqdm(files):
    granule_name = f.split('/')[-1][9:22]
    granule_fname = [f for f in granule_fnames if granule_name in f][0]
    x_ind = int(f.split('_')[4])
    y_ind = int(f.split('_')[6])
    lrim = np.array(Image.open(f))[...,0]
    hrim = np.array(Image.open(granule_fname))[-x_ind*2-257:-x_ind*2-1,-y_ind*2-257:-y_ind*2-1]
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(lrim)
    plt.subplot(1,2,2)
    plt.imshow(np.rot90(hrim,k=2))
    plt.show()
    im = Image.fromarray(hrim)
    im.save('./data/yuan2020/chips/' + f[26:])