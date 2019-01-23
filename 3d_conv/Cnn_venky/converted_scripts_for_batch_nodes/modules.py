### List of modules used for training 

import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from models import f_define_model

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


def f_split_data(inpx,inpy,wts,test_fraction):
    '''
    Split data for training and test. validation from training piece of data.
    !! Warning this code deletes inpx,inpy inside the function. can't help it because the arrays are too big!!
    '''
    
    num=inpx.shape[0]
    test_idx=int(test_fraction*num)
    train_idx=num-test_idx

    train_x,train_y,train_wts=inpx[:train_idx],inpy[:train_idx],wts[:train_idx]
    test_x,test_y,test_wts=inpx[train_idx:],inpy[train_idx:],wts[train_idx:]
    
def f_plot_yinput(inpy,model_dict,title_suffix,save_loc=''):
    # Plot data
    fig_fname='y-input_model-%s.pdf'%(model_dict['name'])
    plt.figure()
    plt.plot(inpy[:],linestyle='',marker='*',markersize=1)
    plt.title("Plot of y data after shuffle: %s "%(title_suffix))
    plt.savefig(save_loc+fig_fname)
    plt.close()

    
def f_format_data(inpx,inpy,wts,shuffle_flag=True,drop_data=True,data_size=1000,test_fraction=0.25):
    ''' Shuffle, drop and split data for train-test
    '''
    # Shuffle data
    if shuffle_flag: inpx,inpy,wts=f_shuffle_data(inpx,inpy,wts)
    # Drop data
    if drop_data: inpx,inpy,wts=f_drop_data(inpx,inpy,wts,data_size)

#     print(inpy[inpy==0.0].shape,inpy[inpy>0.0].shape,inpy.shape)
    
    # Split data into train-test.
    train_x,train_y,train_wts,test_x,test_y,test_wts=f_split_data(inpx,inpy,wts,test_fraction)
    
    print('Data sizes: train_x{0},train_y{1},test_x{2},test_y{3}'.format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))

    return train_x,train_y,train_wts,test_x,test_y,test_wts


def f_train_model(model,inpx,inpy,num_epochs=5):
    '''
    Train model. Returns just history.history
    '''
    cv_fraction=0.33 # Fraction of data for cross validation
    
    history=model.fit(x=inpx, y=inpy,
                    batch_size=32,
                    epochs=num_epochs,
                    verbose=1,
#                     callbacks = [callbacks.ModelCheckpoint('./rpv_weights.h5')],
                    validation_split=cv_fraction,
                    shuffle=True
                )
    
    print("Number of parameters",model.count_params())
    
    return history.history


def f_plot_learning(history,model_name,save_loc=''):
    ''' Plot learning curves'''
    
    fig_name='learning_model%s.pdf'%(model_name)

    fig=plt.figure()
    # Plot training & validation accuracy values
    fig.add_subplot(2,1,1)
    plt.plot(history['acc'],label='Train')
    plt.plot(history['val_acc'],label='Validation')
#     plt.title('Model accuracy')
    plt.ylabel('Accuracy')

    # Plot loss values
    fig.add_subplot(2,1,2)
    plt.plot(history['loss'],label='Train')
    plt.plot(history['val_loss'],label='Validation')
#     plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')

    plt.savefig(save_loc+fig_name)
    plt.close()

def f_plot_roc_curve(fpr,tpr,model_name,save_loc=''):
    '''
    Module for roc plot and printing AUC
    '''
    fig_name='roc_curve_model%s.pdf'%(model_name)
    plt.figure()
    plt.scatter(fpr,tpr)
    
    plt.semilogx(fpr, tpr)
#     Zooms
    plt.xlim([10**-7,1.0])
    plt.ylim([0,1.0])
    
    # AUC 
    auc_val = auc(fpr, tpr)
    print("AUC: ",auc_val)
    
    plt.savefig(save_loc+fig_name)
    plt.close()

    

def f_test_model(xdata,ydata,wts,model,model_name,model_save_dir,test_status=False):
    '''
    Test model and make ROC plot
    If model has been tested, store the y-predict values, test_y values and test_weight values
    and read them in next time.
    '''
    
    test_file_name=model_save_dir+'y-predict_model-'+str(model_name)+'.pred'
    test_y_file_name=model_save_dir+'y-test_model-'+str(model_name)+'.test'
    test_weights_file_name=model_save_dir+'wts-test_model-'+str(model_name)+'.test'
    
    
#     model.evaluate(xdata,ydata,sample_weights=wts,verbose=1)
    if not test_status:# Predict values and store to file.
        y_pred=model.predict(xdata,verbose=1)
        # Save prediction file
        np.savetxt(test_file_name,y_pred)
        np.savetxt(test_y_file_name,ydata)
        np.savetxt(test_weights_file_name,wts)
            
    else: # Load y_predictions from file. Note: This overwrites the ydata and wts arrays with the stored values.
        print("Using test prediction from previous test",test_file_name,test_y_file_name,test_weights_file_name)
        y_pred=np.loadtxt(test_file_name)
        ydata=np.loadtxt(test_y_file_name)
        wts=np.loadtxt(test_weights_file_name)
    
    assert(ydata.shape[0]==y_pred.shape[0]),"Data %s and prediction arrays %s are not of the same size"%(test_y.shape,y_pred.shape)
       
    # For resnet, the output has 2 columns, you pick the second one.
    if y_pred.shape[1]==2: y_pred=y_pred[:,1]
#     print(y_pred)

    fpr,tpr,threshold=roc_curve(ydata,y_pred,sample_weight=wts)
    print(fpr.shape,tpr.shape,threshold.shape)
    # Plot roc curve
    f_plot_roc_curve(fpr,tpr,model_name,save_loc=model_save_dir)


def f_perform_fit(train_x,train_y,train_wts,test_x,test_y,test_wts,model_dict,model_save_dir,num_epochs=5,train_status=False,test_status=False):
    '''
    Compile, train, save and test the model.
    Steps:
    - Compile
    - Train
    - Save
    - Read
    - Plot
    - Test
    
    Note: Cross-validation data is built into the training. So, train_{x/y} contains the training and cval data.
    '''
    
    model_name=model_dict['name'] # string for the model
    fname_model,fname_history='model_{0}.h5'.format(model_name),'history_{0}.pickle'.format(model_name)
    
    # Plot the y-input
    
    f_plot_yinput(train_y,model_dict,title_suffix='train_data',save_loc=model_save_dir) 
    f_plot_yinput(test_y,model_dict,title_suffix='test_data',save_loc=model_save_dir) 
    
    if not train_status: # If not trained before, train the model and save it.

        ########################
        # Compile model
        model=f_define_model(train_x,model_name)
        # Train model
        history=f_train_model(model,train_x,train_y,num_epochs)

        ########################
        # Save model and history
        model.save(model_save_dir+fname_model)
        with open(model_save_dir+fname_history, 'wb') as f:
                pickle.dump(history, f)
    
    else:
        print("Using trained model")

        
    ########################
    ### Read model and history
    
    ### Check if files exist
    assert os.path.exists(model_save_dir+fname_model),"Model not saved"
    assert os.path.exists(model_save_dir+fname_history),"History not saved"
    
    model=load_model(model_save_dir+fname_model)
    with open(model_save_dir+fname_history,'rb') as f:
        history= pickle.load(f)
    
    ########################
    model.summary()
    # Plot tested model
    f_plot_learning(history,model_name,save_loc=model_save_dir)
    
    ########################
    # Test model
    f_test_model(test_x,test_y,test_wts,model,model_dict['name'],model_save_dir,test_status)
    
    model_dict['model'],model_dict['history']=model,history
    
    return model_dict
