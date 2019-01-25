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
    # add_arg('--verbose','-v',   action='store_true',    help='Show extra details of training and testing ')

    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    print(args)
    ## Note: --train means models needs to be trained. hence train_status=False
    train_status,test_status=args.train_status,args.test_status
    #print(train_status,test_status)

    ###Extract data #######
    data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/data_hesse_cuts/'
    ### Extract regular data
    f1,f2,f3='processed_input_regular_x','processed_input_regular_y','processed_input_regular_wts'
    ix,iy,iwts=f_load_data(data_dir,f1,f2,f3,mode=True)
    print("Num samples in regular data",iy.shape[0])
    print(ix.shape,type(ix),type(iy[0]))
    print(sys.getsizeof(ix),sys.getsizeof(iy))
    
    t1=time.time()
    i1x,i1y,i1wts=f_shuffle_data(ix,iy,iwts)
    t2=time.time()
    print('After shuffle 1',t2-t1)
    print(sys.getsizeof(i1x),sys.getsizeof(i1y))
           
    ### Extract reserved data
    f1,f2,f3='processed_input_reserved_x','processed_input_reserved_y','processed_input_reserved_wts'
    ix,iy,iwts=f_load_data(data_dir,f1,f2,f3,mode=True)
    print("Num samples in reserved data",iy.shape[0])

    t1=time.time()
    i2x,i2y,i2wts=f_shuffle_data(ix,iy,iwts)
    t2=time.time()
    print('After shuffle 2',t2-t1)
    print(sys.getsizeof(i2x),sys.getsizeof(i2y))

    raise SystemExit


    ###Format data #######
    # --- cross-validation done with part of train data.
    ## Combine regular and reserved data
    #inpx,inpy,wts=np.vstack([i1x,i2x]),np.hstack([i1y,i2y]),np.hstack([i1wts,i2wts])

    ###Shuffle data
    t1=time.time()
    i1x,i1y,i1wts=f_shuffle_data(i1x,i1y,i1wts)
    t2=time.time()
    print('After shuffle 1',t2-t1)
    ###Split data
    #size=inpy.shape[0]
    #train_idx=int(size*0.75)
    #train_x,train_y,train_wts=inpx[:train_idx],inpy[:train_idx],wts[:train_idx]
    #test_x,test_y,test_wts=inpx[train_idx:],inpy[train_idx:],wts[train_idx:]

    train_x,train_y,train_wts=i1x.copy(),i1y.copy(),i1wts.copy()
    t3=time.time()
    print('Time after copy',t3-t2)
    del(i1x,i1y,i1wts)
    # Shuffling test data set because of the split done.
    t4=time.time()
    i2x,i2y,i2wts=f_shuffle_data(i2x,i2y,i2wts)
    t5=time.time()
    print('After shuffle 2',t5-t4)
    res_size=i2y.shape[0]
    split_size=int(res_size/2)
    test_x,test_y,test_wts=i2x[:split_size],i2y[:split_size],i2wts[:split_size]
    t6=time.time()
    print("Test data size",test_y.shape[0])
    print(t6-t5)

    #test_x,test_y,test_wts=i2x,i2y,i2wts
    del(i2x,i2y,i2wts)

