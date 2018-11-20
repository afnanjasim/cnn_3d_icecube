import sys
import os

import numpy as np
import glob
import pickle


# Import modules form other nbs
from nbfinder import NotebookFinder
sys.meta_path.append(NotebookFinder())
from load_data import get_data
#from resnet_util import identity_block,conv_block



#################################
### Code starts here

### Read Data from files ###
sigpath = "/project/projectdirs/dasrepo/icecube_data/hdf5_out/filtered/nugen/11374/clsim-base-4.0.3.0.99_eff/"
# sig_list=glob.glob(sigpath+'*00.hdf5') # If you want to play with a smaller data set.
sig_list=glob.glob(sigpath+'*00.hdf5')
bgpath = "/project/projectdirs/dasrepo/icecube_data/hdf5_out/filtered/corsika/11057/"
bg_list=glob.glob(bgpath+'*00.hdf5')

inx,inpy = get_data(sig_list, bg_list)
num=inx.shape[0]
print("Data shape after read:\tx:{0}\ty:{1}".format(inx.shape,inpy.shape))

### Format data for keras 3D CNN ###
inx2=np.expand_dims(inx,axis=1)
inx3=np.transpose(inx2,axes=[0,3,4,2,1])
# print(inx2.shape,inx3.shape)

inpx=inx3.copy()
print("Data shape after format:\tx:{0}\ty:{1}".format(inpx.shape,inpy.shape))

### Write data to files ###
# Save data to files
f1='processed_input_x'
f2='processed_input_y'
#np.save(f1,inpx)
#np.save(f2,inpy)


### Done ###
