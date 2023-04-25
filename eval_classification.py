from PIL import Image
from glob import glob
import numpy as np
from utils import confusion, TC_to_reflectance, to_unit_vecs, cos_dist
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from neural_networks import res_net, testing_network, pretrained
from tqdm import tqdm
mode= 'scratch'#RN50','DN121','scratch','goes_bt','bt'
dataset= 'rasp2020' #'yuan2020', 'rasp2020'

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

#prep the model:
if mode == 'scratch':
    encoder = res_net()
elif mode == 'RN50' or mode == 'DN121':
    encoder = pretrained(256,mode)
elif mode == 'bt':
    encoder = load_model('./data/cnns/modis_bt/res_net_encoder_epoch1000')
elif mode == 'goes_bt':
    encoder = load_model('./data/cnns/goes_bt/res_net_encoder_epoch1000')
if mode != 'scratch':
    encoder.trainable=False
cnn = testing_network(encoder, 256, len(types))
cnn.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics='categorical_accuracy')

#train the model:
es_callback = EarlyStopping(patience=10,verbose=1,restore_best_weights=True,monitor='val_categorical_accuracy',mode='max')
cnn.fit(inputs_train,targets_train,verbose=1,batch_size=64,epochs=1000,validation_data=(inputs_valid,targets_valid),callbacks=[es_callback])
K.set_value(cnn.optimizer.learning_rate, 0.0001)
cnn.fit(inputs_train,targets_train,verbose=1,batch_size=64,epochs=1000,validation_data=(inputs_valid,targets_valid),callbacks=[es_callback])
test_outputs = cnn.predict(inputs_test,verbose=False).squeeze()
test_labels = np.argmax(targets_test,axis=-1)
pred = np.argmax(test_outputs,axis=-1)
acc, conf = confusion(test_labels,pred,types)
save_file = './data/' + dataset + '/evaluations/' + mode + '.npz'
np.savez(save_file, accuracy = acc, confusion = conf)