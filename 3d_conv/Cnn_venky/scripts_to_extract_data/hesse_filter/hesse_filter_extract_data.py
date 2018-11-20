#!/usr/bin/env python
# coding: utf-8

# ## Code to extract training and testing data from hdf5 files and storing them in the right form in .npy files
# 
# This script gives processed data reading.
# Only dependency is util.py
# - Nov 12, 2018

import sys
import os

import numpy as np
import glob
import pickle
import h5py

import time


# In[ ]:


# Import modules from other files
from util import add_pulse_to_inp_tensor, get_nonempty_pulses, total_doms, total_height, total_width, get_pulse_array, get_nonempty_events


# ### Modules to make dataset

# In[41]:



def make_dataset(filename, sig_or_bg,cut=None):
    '''
    Create arrays for xinput, yinput and weights from a single file name
    '''
    ####### Modified by Venkitesh, Nov 19, 2018. 
    
    hf = h5py.File(filename,'r')
    
    pulse_array_keys = get_nonempty_pulses(hf)
    event_array_keys=get_nonempty_events(hf)
    # Checking whether the event_array_keys and pulse_array_keys are in order and identical
    assert len(pulse_array_keys)==len(event_array_keys), "Pulse and event array keys have different sizes"
    assert np.array_equal(pulse_array_keys,event_array_keys), "Pulse array %s and Event array %s are not identical. Possibility of mismatch"%(pulse_array_keys,event_array_keys)
    
    if (sig_or_bg=='sig' and cut=='hesse'):
        key_lst=[] # List that will store the events that satisfy the cuts.
        for evt in event_array_keys:
            val=hf['events'][evt]
            if ((np.abs(val['true_x'][0])<500) and (np.abs(val['true_y'][0])<500) and (np.abs(val['true_z'][0])<500)) :
                print("Hesse-cut",filename)
                print(val,val['true_x'][0],val['true_y'][0],val['true_z'][0])
                key_lst.append(evt)
        array_keys=np.array(key_lst)
    else: 
        array_keys=event_array_keys.copy()
    
    num_events = len(array_keys)
    # Computing the weights
    wgts=np.array([hf['events'][event_key]['weight'][0] for event_key in array_keys])
            
    tens = np.zeros((num_events, total_doms, total_height, total_width))
    for ex_num, pulse_array_key in enumerate(array_keys):
        pulse_array = get_pulse_array(hf, pulse_array_key)
        add_pulse_to_inp_tensor(tens, ex_num, pulse_array)
        
    lbls = np.ones((num_events,)) if sig_or_bg == "sig" else np.zeros((num_events,))
        
    return tens, lbls, wgts


def get_data(filename_list,file_type,cut):
    ''' file_type="sig" or "bg" '''
    
    assert (file_type=="sig" or file_type=="bg"), "invalid file_type %s: must be sig or bg"%(file_type)
    # Create first row of numpy array
    x, y, wt = make_dataset(filename_list[0], file_type)
    # Then append to it
    for fn in filename_list[1:]:
        xs,ys,wts = make_dataset(fn, file_type,cut)
        x = np.vstack((x,xs))
        y = np.concatenate((y,ys))
        wt = np.concatenate((wt,wts))
    
    return x,y,wt


# In[42]:



def f_get_file_lists(data_folder,mode):
    ''' Function to the get the list of signal files and background files (sigpath and bgpath) for reserved and training data. 
        mode='quick' picks a smaller set of files for quick training. These files have the form '*00.hdf5'.
        
        Arguments:
        data_folder='regular' or 'reserved'
        mode='regular' or 'quick'
    '''
    
    if data_folder=='reserved':
        sigpath = "/project/projectdirs/dasrepo/icecube_data/reserved_data/filtered/nugen/11374/clsim-base-4.0.3.0.99_eff/"
        bgpath = "/global/project/projectdirs/dasrepo/icecube_data/reserved_data/filtered/corsika/11057/"
    else:
        sigpath = "/project/projectdirs/dasrepo/icecube_data/hdf5_out/filtered/nugen/11374/clsim-base-4.0.3.0.99_eff/"
        bgpath = "/project/projectdirs/dasrepo/icecube_data/hdf5_out/filtered/corsika/11057/"
        
    
    # For quick testing, use only the file starting with a '00' at the end ('*00.hdf5'). This give a much smaller set of files, for quick testing.
    suffix='*00.hdf5' if mode=='quick' else '*.hdf5'     
    sig_list=glob.glob(sigpath+suffix)
    bg_list=glob.glob(bgpath+suffix)
    
    return sig_list,bg_list


def f_extract_data(data_folder,save_location,mode='normal',cut=None):
    '''
    Function to perform :
    - Data read
    - Data format
    - Data save to file
    
    Arguments:
    data_folder='regular' or 'reserved'
    save_location= location to save the data files (that are very large)
    mode='normal' or 'quick'
    '''
    
    
    def f_concat_temp_files():
        ''' get data from temp files, stack numpy array and delete temp files'''
        for i in np.arange(count):
            prefix='temp_data_%s'%(i)
            f1,f2,f3=[prefix+i+'.npy' for i in ['_x','_y','_wts']]
            xs,ys,wts=np.load(save_location+f1),np.load(save_location+f2),np.load(save_location+f3)

            if i==0:
                x=xs;y=ys;wt=wts
            else:
                x = np.vstack((x,xs))
                y = np.concatenate((y,ys))
                wt = np.concatenate((wt,wts))

            for fname in [f1,f2,f3]: os.remove(save_location+fname) # Delete temp file
        return x,y,wt
    
    
    print("Type of data:\t",data_folder)
    
    ##########################################
    ### Read Data from files ###
    sig_list,bg_list=f_get_file_lists(data_folder,mode)
    print(len(sig_list),len(bg_list))
    
    count=0 # counter for index of temp file
    for file_list,sig_or_bg in zip([sig_list,bg_list],['sig','bg']):
        print('Type: ',sig_or_bg)
        num_files=len(file_list); block_size=100
        num_blocks=int(num_files/block_size)+1
        print("Number of blocks",num_blocks)
        for i in np.arange(num_blocks):
            t1=time.time()
            start=i*block_size
            end=None if i==(num_blocks-1) else (i+1)*block_size # exception handling for last block
            
            f_list=file_list[start:end]
            inx,inpy,wts = get_data(f_list,sig_or_bg,cut)
            
            ### Save data for each block to temp files ###
            prefix='temp_data_%s'%(count)
            f1,f2,f3=prefix+'_x',prefix+'_y',prefix+'_wts'
            for fname,data in zip([f1,f2,f3],[inx,inpy,wts]):
                np.save(save_location+fname,data)
            
            count+=1 # count is updated for both signal and bgnd
            t2=time.time()
            print("block number: ",i,"Start-End",start,end,"  time taken in seconds: ",t2-t1)
        
        print("Number of samples after %s: %s "%(sig_or_bg,inpy.shape[0]))
    print("Number of temp files written",count)
    
    # concatenate files to get full input data files
    t1=time.time()
    inx,inpy,wts=f_concat_temp_files()
    t2=time.time()
    print("Time taken for concatenating temp files",t2-t1)
    num=inx.shape[0]
    print("Data shape after read:\tx:{0}\ty:{1}\twts:{2}".format(inx.shape,inpy.shape,wts.shape))
    
    ##########################################
    ### Format the x-data for keras 3D CNN ###
    inx2=np.expand_dims(inx,axis=1)
    inx3=np.transpose(inx2,axes=[0,3,4,2,1])
    # print(inx2.shape,inx3.shape)
    inpx=inx3.copy()
    print("Data shape after format:\tx:{0}\ty:{1}".format(inpx.shape,inpy.shape,wts.shape))
    
    ##########################################
    ### Save data to files ###
    prefix='processed_input_'+data_folder
    f1,f2,f3=prefix+'_x',prefix+'_y',prefix+'_wts'

    for fname,data in zip([f1,f2,f3],[inpx,inpy,wts]):
        np.save(save_location+fname,data)
        


# In[ ]:


if __name__=='__main__':
    
    save_data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/'
    ### Regular data
    t1=time.time()
#     f_extract_data(data_folder='regular',save_location=save_data_dir,mode='quick',cut='hesse')
    f_extract_data(data_folder='regular',save_location=save_data_dir,mode='normal',cut='hesse')

    t2=time.time()
    print("Time taken in minutes ",(t2-t1)/60.0)

    ### Reserved data ###
    t1=time.time()
#     f_extract_data(data_folder='reserved',save_location=save_data_dir,mode='quick',cut='hesse')
    f_extract_data(data_folder='reserved',save_location=save_data_dir,mode='normal',cut='hesse')
    t2=time.time()
    
    print("Time taken in minutes ",(t2-t1)/60.0)
# ## Notes:
# Nov 12, 2018
# 
# Tested this code by doing a diff of regular files with those produced before and they match!
# Test of times for various stages:
# 
# #### Regular data:
# Rough times for each stage in seconds
# Time for signal         2260
# Time for bg             8320
# Time for extraction     323
# 
# Total time in hours:    2.91 hours
# 
# #### Reserved data:
# 
# Rough times for each stage in seconds
# Time for signal         8588
# Time for bg             51300
# Time for extraction     10803
# 
# Total time in hours:    18.45 hours
