#!/usr/bin/env python3
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
import numpy as np
from neural_networks import barlow_model, MLP, res_net
from data_manager import barlow_batch
from tqdm import tqdm
import concurrent.futures
from time import time

model_name = 'res_net'
dir_name = 'modit_bt'
encoder = res_net()
projector = MLP(1024,[2048,4096,8192])

batch_size = 512
blwt = barlow_model(encoder, projector, batch_size)
dummy_targets = np.zeros((batch_size,))
errors = []

def fit(batch):
    return blwt.train_on_batch(batch,dummy_targets)

threader = concurrent.futures.ThreadPoolExecutor(max_workers=2)
batch = barlow_batch(batch_size)
training_loss = []
for epoch in range(1_000):
    epoch_error = []
    t0 = time()
    for i in tqdm(range(100)):
        fit_thread = threader.submit(fit,batch)
        batch_thread = threader.submit(barlow_batch,batch_size)
        batch_loss = fit_thread.result()
        epoch_error.append(batch_loss)
        batch = batch_thread.result()
    print('Epoch ' + str(epoch) + ' loss: ' + str(np.round(np.mean(epoch_error),2)) + '  Elapsed Time: ' + str(np.round(time()-t0)) + 's')
    training_loss.append(np.mean(epoch_error))
    np.save('./data/cnns/' + dir_name + '/loss.npy',training_loss)
    if epoch % 10 == 0:
        encoder.save('./data/cnns/' + dir_name + '/' + model_name + '/encoder_epoch' + str(epoch).zfill(4))
        projector.save('./data/cnns/' + dir_name + '/' + model_name + '/projector_epoch' + str(epoch).zfill(4))