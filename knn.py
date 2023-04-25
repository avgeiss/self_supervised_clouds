from PIL import Image
from glob import glob
import numpy as np
from utils import confusion, TC_to_reflectance, to_unit_vecs, cos_dist
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from neural_networks import res_net, testing_network, pretrained
from tqdm import tqdm
dataset= 'yuan2020' #'yuan2020', 'rasp2020'

if dataset == 'yuan2020':
    types = ['Closed-cellular_MCC','Clustered_cumulus','Disorganized_MCC','Open-cellular_MCC','Solid_Stratus','Suppressed_Cu']
    chip_fnames = sorted(glob('./data/yuan2020/chips/*/*.png'))
elif dataset == 'rasp2020':
    types = ['Sugar','Gravel','Fish','Flower']
    chip_fnames = sorted(glob('./data/rasp2020/chips/*/*.png'))

def load_image(fname):
    im = Image.open(fname)
    im = np.array(im)
    if im.ndim == 3:
        im = TC_to_reflectance(im[:,:,1])
    im = np.stack([im,np.rot90(im,1),np.rot90(im,2),np.rot90(im,3)])
    label = np.zeros((4,len(types)))
    label[:,types.index(fname.split('/')[-2])] = 1
    return im, label

def load_ims_by_idx(fnames, inds):
    images, labels = zip(*[load_image(fnames[i]) for i in tqdm(inds)])
    return np.concatenate(images), np.concatenate(labels)
    
#load the dataset
N = len(chip_fnames)
np.random.seed(0)
shuf_inds = np.arange(N)
np.random.shuffle(shuf_inds)
fnames = np.array(chip_fnames)
train, valid, test = shuf_inds[:int(N*.7)], shuf_inds[int(N*.7):int(N*.8)], shuf_inds[int(N*.8):]
inputs_train, targets_train = load_ims_by_idx(fnames,train)
inputs_valid, targets_valid = load_ims_by_idx(fnames,valid)
inputs_test,  targets_test  = load_ims_by_idx(fnames,test )

#compute embeddings:
cnn = load_model('./data/cnns/modis_bt/res_net_encoder_epoch1000')
trn_emb = cnn.predict(inputs_train,batch_size=128)
val_emb = cnn.predict(inputs_valid,batch_size=128)
tst_emb = cnn.predict(inputs_test,batch_size=128)
trn_emb = to_unit_vecs(np.double(trn_emb))
val_emb = to_unit_vecs(np.double(val_emb))
tst_emb = to_unit_vecs(np.double(tst_emb))

#optimize K with validation set:
max_k = 300
dists = cos_dist(trn_emb, val_emb)
ranks = np.argsort(dists,axis=0)[:max_k,:]
labels = targets_train[ranks,:]
cumprob = np.cumsum(labels,axis=0)/np.arange(1,max_k+1)[:,np.newaxis,np.newaxis]
cumlabels = np.argmax(cumprob,axis=-1)
true_labels = np.argmax(targets_valid,axis=1)
val_acc = np.mean(cumlabels == true_labels[np.newaxis,:],axis=1)
print(np.argmax(val_acc))

k = 40
dists = cos_dist(trn_emb,tst_emb)
ranks = np.argsort(dists,axis=0)[:k,:]
labels = targets_train[ranks,:]
labels = np.mean(labels,axis=0)
labels = np.argmax(labels,axis=-1)
true_labels = np.argmax(targets_test,axis=1)
confusion(true_labels, labels, types)  