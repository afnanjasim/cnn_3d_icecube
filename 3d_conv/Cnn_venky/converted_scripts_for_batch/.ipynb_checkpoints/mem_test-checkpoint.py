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


N=200000
a=np.arange(N)
b=a[:]
print(sys.getsizeof(a),sys.getsizeof(b))

def f_shuffle(a):
    
    # Setting seed
    seed=243
    np.random.seed(seed=seed)

    size=a.shape[0]
    ## Get shuffled array of indices
    shuffle_arr=np.arange(size)
    np.random.shuffle(shuffle_arr)
    inp=a[shuffle_arr]

    return inp

c=f_shuffle(a)
print(c[:10])
print(sys.getsizeof(a),sys.getsizeof(c))
time.sleep(10)


# Setting seed
seed=243
np.random.seed(seed=seed)

size=a.shape[0]
## Get shuffled array of indices
shuffle_arr=np.arange(size)
np.random.shuffle(shuffle_arr)
d=a[shuffle_arr]
# Third way
print(sys.getsizeof(a),sys.getsizeof(b),sys.getsizeof(c),sys.getsizeof(d))


