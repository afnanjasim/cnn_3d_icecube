# Code to train CNN models using keras and tensor flow

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import time
import argparse 
import datetime

## M-L modules
import keras
from keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import load_model

## special imports for setting keras and tensorflow variables.
sys.path.insert(0,'/global/u1/v/vpa/standard_scripts/')
from keras_tf_parallel_variables import configure_session

## modules from other files
from models import *
from modules import *

########################
## Code starts
########################
#Steps: Extract data, process data, train data using model, plot learning, test data with model, plot roc curve.

### Set tensorflow and keras variables
configure_session(intra_threads=32, inter_threads=2, blocktime=1, affinity='granularity=fine,compact,1,0')

### Parsing arguments
parser = argparse.ArgumentParser(description="Train and test CNN for ice-cube data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train','-tr',  action='store_false' ,dest='train_status' ,help='Has the model been trained?')
parser.add_argument('--test', '-ts',  action='store_false' ,dest='test_status'  ,help='Has the model been tested?')
# parser.add_argument('--verbose','-v',   action='store_true',    help='Show extra details of training and testing ')

## Note: --train means models needs to be trained. hence train_status=False
parg = parser.parse_args(sys.argv[1:])
train_status,test_status=parg.train_status,parg.test_status
print(train_status,test_status)

###Extract data #######
data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/'
### Extract regular data
f1,f2,f3='processed_input_regular_x','processed_input_regular_y','processed_input_regular_wts'
i1x,i1y,i1wts=f_load_data(data_dir,f1,f2,f3)
print("Num samples in regular data",i1y.shape[0])
### Extract reserved data
f1,f2,f3='processed_input_reserved_x','processed_input_reserved_y','processed_input_reserved_wts'
i2x,i2y,i2wts=f_load_data(data_dir,f1,f2,f3)
print("Num samples in reserved data",i2y.shape[0])

###Format data #######
# --- cross-validation done with part of train data.
## Combine regular and reserved data
#inpx,inpy,wts=np.vstack([i1x,i2x]),np.hstack([i1y,i2y]),np.hstack([i1wts,i2wts])
inpx,inpy,wts=i1x,i1y,i1wts
del(i1x,i1y,i1wts)

###Shuffle data
inpx,inpy,wts=f_shuffle_data(inpx,inpy,wts)
###Split data
#size=inpy.shape[0]
#train_idx=int(size*0.75)
#train_x,train_y,train_wts=inpx[:train_idx],inpy[:train_idx],wts[:train_idx]
#test_x,test_y,test_wts=inpx[train_idx:],inpy[train_idx:],wts[train_idx:]

train_x,train_y,train_wts=inpx,inpy,wts
# Not shuffling data for test
test_x,test_y,test_wts=i2x,i2y,i2wts
del(inpx,inpy,wts)
del(i2x,i2y,i2wts)

### Train and test model
# All models in sequence:
model_save_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/saved_models/'
#for i in range(1,3):
for i in range(1,6):
    print(i,'{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    model_dict={'name':str(i),'description':'-','model':None,'history':None}
    num_epochs=20
    model_dict1=f_perform_fit(train_x,train_y,train_wts,test_x,test_y,test_wts,model_dict,model_save_dir,num_epochs,train_status,test_status)

