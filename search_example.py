from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils import to_unit_vecs, cos_dist, TC_to_reflectance, reflectance_to_TC
from PIL import Image
from glob import glob

def load_modis_chip_by_idx(idx):
    fnames = sorted(glob('./data/modis/training_images/*.npy'))
    f = fnames[idx//16]
    image = np.load(f)
    i = (idx % 16) % 4
    j = (idx % 16 ) // 4
    chip = image[i*256:(i+1)*256,j*256:(j+1)*256]
    return chip

cnn = load_model('./data/cnns/modis_bt/res_net_encoder_epoch1000')
modis_embeddings = np.load('./data/modis/embeddings.npy')
modis_embeddings = to_unit_vecs(np.double(modis_embeddings))

types = ['Flower','Fish','Gravel','Sugar']
sample_files = ['Aqua_R3_20131012_994_1642.png',
                'Aqua_R1_20120426_1107_409.png',
                'Aqua_R1_20140204_15_18.png',
                'Aqua_R1_20070308_562_1588.png']
rows = []
for t, sf in zip(types,sample_files):
    sample = np.array(Image.open('./data/rasp2020/chips/' + t + '/' + sf))[:,:,1]
    sample = TC_to_reflectance(sample)
    row = [sample]
    e = cnn(sample[np.newaxis,:,:,np.newaxis],training=False)
    e = to_unit_vecs(np.double(e))
    dists = cos_dist(modis_embeddings,e).squeeze()
    closest_inds = np.argsort(dists)[0:5]
    for i in range(5):
        chip = load_modis_chip_by_idx(closest_inds[i])
        chip -= int(np.min(chip)*0.5)
        row.append(chip)
    rows.append(np.concatenate(row,axis=-1))
composite = np.concatenate(rows,axis=0)
composite = reflectance_to_TC(composite)

for i in range(7):
    n = i*256
    composite[:,n-2:n+2] = 255
for i in range(5):
    n = i*256
    composite[n-2:n+2,:] = 255

plt.figure(figsize=(12,8))
plt.imshow(composite,cmap='bone',vmin=0,vmax=255)
plt.xticks(np.arange(128,256*6+128,256),labels=['Query Image','Sample 1','Sample 2','Sample 3','Sample 4','Sample 5'])
plt.yticks(np.arange(128,256*4+128,256),labels=types,rotation=90,va='center')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('./figures/sample_search.png', dpi=300, bbox_inches='tight')

types = ['Closed-cellular_MCC','Clustered_cumulus','Disorganized_MCC','Open-cellular_MCC','Solid_Stratus','Suppressed_Cu']
type_titles = ['Closed Cellular','Clustered Cu','Disorganized','Open Cellular','Solid Stratus','Suppressed Cu']
sample_files = ['MYD021KM.A2010157.2245_index_0512_index_0640_Ref7.png',
                'MYD021KM.A2010211.2205_index_0384_index_0768_Ref7.png',
                'MYD021KM.A2010176.2315_index_0256_index_0768_Ref7.png',
                'MYD021KM.A2010173.1920_index_1792_index_0512_Ref7.png',
                'MYD021KM.A2010180.2115_index_1024_index_0896_Ref7.png',
                'MYD021KM.A2010154.2030_index_1408_index_0512_Ref7.png']
rows = []
for t, sf in zip(types,sample_files):
    sample = np.array(Image.open('./data/yuan2020/chips/' + t + '/' + sf))
    e = cnn(sample[np.newaxis,:,:,np.newaxis],training=False)
    sample[:2,:] = 255;sample[-2:,:] = 255;sample[:,:2] = 255;sample[:,-2:] = 255
    row = [sample]
    e = to_unit_vecs(np.double(e))
    dists = cos_dist(modis_embeddings,e).squeeze()
    closest_inds = np.argsort(dists)[0:5]
    for i in range(5):
        chip = load_modis_chip_by_idx(closest_inds[i])
        chip[:2,:] = 255
        chip[-2:,:] = 255
        chip[:,:2] = 255
        chip[:,-2:] = 255
        row.append(chip)
    rows.append(np.concatenate(row,axis=-1))
composite = np.concatenate(rows,axis=0)

plt.figure(figsize=(12,12))
plt.imshow(composite,cmap='bone',vmin=0,vmax=255)
plt.xticks(np.arange(128,256*6+128,256),labels=['Query Image','Sample 1','Sample 2','Sample 3','Sample 4','Sample 5'])
plt.yticks(np.arange(128,256*6+128,256),labels=type_titles,rotation=90,va='center')
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.savefig('./figures/sample_search_yuan2020.png', dpi=300, bbox_inches='tight')
