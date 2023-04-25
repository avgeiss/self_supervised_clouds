# build_goes_dataset.py

#find the files
from glob import glob
import numpy as np
from netCDF4 import Dataset
import datetime
from numpy import cos, sin, arctan, sqrt, pi
from global_land_mask import globe
from multiprocess import Pool

def land_mask(lat,lon):
    nan_locs = np.isnan(lat+lon)
    lon[lon<-180] += 360
    lat[nan_locs] = 0
    lon[nan_locs] = 0
    mask = globe.is_land(lat,lon)
    mask[nan_locs] = True
    return mask

def proc_goes_file(fname):
    dataset = Dataset(fname)
    
    #determine the solar zenith angle grid
    x = np.double(dataset.variables['x'][:].data)
    y = np.double(dataset.variables['y'][:].data)
    E_sun = np.double(dataset.variables['esun'][:].data)
    d = np.double(dataset.variables['earth_sun_distance_anomaly_in_AU'][:].data)
    proj = dataset.variables['goes_imager_projection']
    r_eq = proj.semi_major_axis
    r_pol = proj.semi_minor_axis
    lambda_0 = proj.longitude_of_projection_origin*pi/180
    H = r_eq + proj.perspective_point_height
    radiances = np.double(dataset.variables['Rad'][:].data)
    radiances[radiances == 4095] = np.nan
    
    #break up the image into chunks:
    N = 1024
    count = 0
    base_fname = './data/goes/training_images/' + fname.split('/')[-1].split('_e2')[0]
    for i in range(0,radiances.shape[0]-N,N):
        for j in range(0,radiances.shape[1]-N,N):
            
            #extract the radiances:
            rad = radiances[i:i+N,j:j+N]
            
            #check if there is missing data:
            if np.any(np.isnan(rad)):
                continue
            
            #get the lat lon grid:
            xx,yy = np.meshgrid(x[j:j+N],y[i:i+N])
            a = sin(xx)**2 + (cos(xx)**2)*(cos(yy)**2 + ((r_eq/r_pol)**2)*sin(yy)**2)
            b = -2*H*cos(xx)*cos(yy)
            c = H**2 - r_eq**2
            r_s = (-b-sqrt(b**2-4*a*c))/(2*a)
            s_x = r_s*cos(xx)*cos(yy)
            s_y = -r_s*sin(xx)
            s_z = r_s*cos(xx)*sin(yy)
            longitude = (lambda_0 - arctan(s_y/(H-s_x)))*180/pi
            latitude = (arctan((s_z/sqrt((H-s_x)**2 + s_y**2))*(r_eq/r_pol)**2))*180/pi
            
            #check if over ocean:
            if np.mean(land_mask(latitude,longitude)) > 0.1:
                continue
            
            #calculate the solar zenith angle:
            time = np.double(dataset.variables['t'][:].data)
            time = datetime.datetime(2000,1,1,12,0,0) + datetime.timedelta(seconds=time)
            slon = -360*(time.hour/24 + time.minute/(24*60) + time.second/(24*60*60) - 0.5)
            hour_angle = (pi/180)*(longitude-slon)
            day_of_year = int(time.strftime('%j'))
            declination = -23.44 * cos(2*pi*(day_of_year+10)/365) * (pi/180)
            cos_sza = sin(latitude*pi/180)*sin(declination) + cos(latitude*pi/180)*cos(declination)*cos(hour_angle)
            sza = np.arccos(cos_sza)*180/pi
            
            #check for sufficient illumination:
            if np.mean(sza) > 60:
                continue
            
            #do conversion to reflectance:
            reflectance = (rad*pi*d**2)/(E_sun*cos_sza)
            reflectance = reflectance.clip(0,1)
            
            #save the image:
            im = reflectance*255
            np.save(base_fname + '_' + str(i).zfill(5) + '_' + str(j).zfill(5) + '.npy',np.uint8(im))
            count += 1
            
    print(str(count) + ' chips extracted from: ' + fname, flush=True)

nc_files = sorted(glob('./data/goes/nc/*.nc'))
p = Pool(12)
p.map(proc_goes_file,nc_files)
p.close()