# Code that reads processed data and shuffles it and stores it back.
### This is done to avoid memory overload on compute nodes.
### Run this on login nodes after data extraction and before training.



import numpy as np


# Load data from files
def f_load_data(data_dir,f1,f2,f3,mode=False):
    ''' Load extracted data from files. Three files for xdata,ydata,weights.
    arguments: data directory, f1,f2,f3 
    returns : inpx,inpy,weights as arrays
    '''
    m='r' if mode else None
    inpx=np.load(data_dir+f1+'.npy',mmap_mode=m)
    inpy=np.load(data_dir+f2+'.npy',mmap_mode=m)
    wts=np.load(data_dir+f3+'.npy',mmap_mode=m)
    print(inpx.shape,inpy.shape)
    
    return inpx,inpy,wts


#### Shuffle and split data ####

def f_shuffle_data(inpx,inpy,wts):
    ## Shuffle data
    
    # Setting seed
    seed=243
    np.random.seed(seed=seed)

    size=inpx.shape[0]
    ## Get shuffled array of indices
    shuffle_arr=np.arange(size)
    np.random.shuffle(shuffle_arr)
    inpx=inpx[shuffle_arr]
    inpy=inpy[shuffle_arr]
    wts=wts[shuffle_arr]

    return inpx,inpy,wts


if __name__=='__main__':

    data_dir='/global/project/projectdirs/dasrepo/vpa/ice_cube/data_for_cnn/extracted_data_v/data/data_hesse_cuts_dec_25/'
    save_location=data_dir
    
    ### Extract regular data
    f1,f2,f3='processed_input_regular_x','processed_input_regular_y','processed_input_regular_wts'
    ix,iy,iwts=f_load_data(data_dir,f1,f2,f3)
    print("Num samples in regular data",iy.shape[0])
    ### Shuffle data ###
    i1x,i1y,i1wts=f_shuffle_data(ix,iy,iwts)
    ### Save shuffled data ###
    f1,f2,f3='shuffled_input_regular_x','shuffled_input_regular_y','shuffled_input_regular_wts'
    for fname,data in zip([f1,f2,f3],[i1x,i1y,i1wts]):
        np.save(save_location+fname,data)

    del(i1x,i1y,i1wts)
    
    ### Extract reserved data
    f1,f2,f3='processed_input_reserved_x','processed_input_reserved_y','processed_input_reserved_wts'    
    ix,iy,iwts=f_load_data(data_dir,f1,f2,f3)
    print("Num samples in reserved data",iy.shape[0])
    ### Shuffle data ###
    i2x,i2y,i2wts=f_shuffle_data(ix,iy,iwts)
    ### Save shuffled data ###
    f1,f2,f3='shuffled_input_reserved_x','shuffled_input_reserved_y','shuffled_input_reserved_wts'
    for fname,data in zip([f1,f2,f3],[i2x,i2y,i2wts]):
        np.save(save_location+fname,data)
    
    
    
    
    
    
    
    