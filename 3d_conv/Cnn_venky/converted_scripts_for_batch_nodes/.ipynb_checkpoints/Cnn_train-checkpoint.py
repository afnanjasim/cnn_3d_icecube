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
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for ice-cube data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--train','-tr',  action='store_false' ,dest='train_status' ,help='Has the model been trained?')
    add_arg('--test', '-ts',  action='store_false' ,dest='test_status'  ,help='Has the model been tested?')
    add_arg('--typeofdata' , choices=['regular','hesse_cut'] ,default='regular' ,dest='type_of_data' ,help='Is the input data hesse cut or regular ?')

    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    print(args)
    ## Note: --train means models needs to be trained. hence train_status=False
    train_status,test_status=args.train_status,args.test_status
    #print(train_status,test_status)

    ###Extract data #######
    if type_of_data=='hesse_cut':
        data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/data_hesse_cuts/'
    elif type_of_data=='regular':
        data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/data_regular/'

    print("Extracting files from",data_dir)
    
    ### Extract regular data
    f1,f2,f3='shuffled_input_regular_x','shuffled_input_regular_y','shuffled_input_regular_wts'
    i1x,i1y,i1wts=f_load_data(data_dir,f1,f2,f3)
    print("Num samples in regular data",i1y.shape[0])

    ### Extract reserved data
    f1,f2,f3='shuffled_input_reserved_x','shuffled_input_reserved_y','shuffled_input_reserved_wts'
    i2x,i2y,i2wts=f_load_data(data_dir,f1,f2,f3)
    print("Num samples in reserved data",i2y.shape[0])
    
    ###Format data #######
    # --- cross-validation done with part of train data.
    ## Combine regular and reserved data
    #inpx,inpy,wts=np.vstack([i1x,i2x]),np.hstack([i1y,i2y]),np.hstack([i1wts,i2wts])

    train_x,train_y,train_wts=i1x.copy(),i1y.copy(),i1wts.copy()

    # Using only part of test data
    res_size=i2y.shape[0]
    split_size=int(res_size/2)
    
    test_x,test_y,test_wts=i2x[:split_size],i2y[:split_size],i2wts[:split_size]
    print("Test data size",test_y.shape[0])
    #test_x,test_y,test_wts=i2x,i2y,i2wts
    del(i2x,i2y,i2wts)

    ### Train and test model
    # All models in sequence:
    model_save_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/saved_models/'
    #for i in range(1,6):
    for i in range(6,11):
        print(i,'{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        model_dict={'name':str(i),'description':'-','model':None,'history':None}
        num_epochs=20
    model_dict1=f_perform_fit(train_x,train_y,train_wts,test_x,test_y,test_wts,model_dict,model_save_dir,num_epochs,train_status,test_status)



