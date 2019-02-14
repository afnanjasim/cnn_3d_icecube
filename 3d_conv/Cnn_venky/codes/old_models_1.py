## modify this script to change the model.
### Add models with a new index.

from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

### Import the modules for resnet50
from resnet50 import *
from resnet18 import *

### Defining all the models tried in the study
def f_define_model(inpx,name):
    '''
    Function that defines the model and compiles it.
    '''
    
    inputs = layers.Input(shape=inpx.shape[1:])
    h = inputs
    
    # Choose model
    if name=='1':
        # Convolutional layers
        conv_sizes=[10, 10, 10]
        conv_args = dict(kernel_size=(3, 3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(2, 2, 2))(h)
    #         h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(10, activation='relu')(h)
        #    h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
    
    elif name=='2':
        # Convolutional layers
        conv_sizes=[10,10,10]
        conv_args = dict(kernel_size=(3, 3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(2, 2, 2))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='3':
        # Convolutional layers
        conv_sizes=[6,6,6]
        conv_args = dict(kernel_size=(3, 3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(2, 2, 2))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
    
    elif name=='4':
        # Convolutional layers
        conv_sizes=[6,6,6]
        conv_args = dict(kernel_size=(3, 3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(2, 2, 2))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='5':
        # Convolutional layers
        conv_sizes=[6,6]
        conv_args = dict(kernel_size=(2, 4, 15), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(3, 3, 3))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)

    elif name=='6':
        # Convolutional layers
        conv_sizes=[20,20,20,20]
        conv_args = dict(kernel_size=(2, 2, 2), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 1, 2))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='7':
        # Convolutional layers
        conv_sizes=[20,20,20,20]
        conv_args = dict(kernel_size=(2, 2, 2), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 2))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='8':
        # Convolutional layers
        conv_sizes=[20,20,20]
        conv_args = dict(kernel_size=(3, 3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)  
        
    elif name=='9':
        # Convolutional layers
        conv_sizes=[20,20,20]
        conv_args = dict(kernel_size=(2, 2, 2), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h) 
        
    elif name=='10':
        # Convolutional layers
        conv_sizes=[40,40,40]
        conv_args = dict(kernel_size=(2, 4, 12), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)   

    elif name=='11':
        # Convolutional layers
        conv_sizes=[40,40,40]
        conv_args = dict(kernel_size=(2, 4, 12), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        #h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)   

    elif name=='12':
        # Convolutional layers
        conv_sizes=[40,40,40]
        conv_args = dict(kernel_size=(2, 4, 12), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)

        h = layers.Conv3D(80, **conv_args)(h)
        h = layers.Conv3D(120, **conv_args)(h)
        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        #h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)   

    elif name=='13':
        # Convolutional layers
        conv_sizes=[20,40,60]
        conv_args = dict(kernel_size=(2, 4, 12), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            h = layers.Dropout(0.5)(h)
        
        h = layers.Conv3D(80, **conv_args)(h)
        h = layers.Conv3D(100, **conv_args)(h)
        h = layers.Conv3D(120, **conv_args)(h)

        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        #h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)   

    elif name=='14':
        # Convolutional layers
        conv_sizes=[20,40,60]
        conv_args = dict(kernel_size=(2, 4, 12), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv3D(conv_size, **conv_args)(h)
            h = layers.MaxPooling3D(pool_size=(1, 2, 3))(h)
            #h = layers.Dropout(0.5)(h)
        
        h = layers.Conv3D(80, **conv_args)(h)
        h = layers.Conv3D(100, **conv_args)(h)
        h = layers.Conv3D(120, **conv_args)(h)

        h = layers.Flatten()(h)
        
        # Fully connected  layers
        h = layers.Dense(120, activation='relu')(h)
        #h = layers.Dropout(0.5)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)   


    elif name=='15': # Resnet 50 
        #from resnet50 import *
        model = ResNet50(img_input=inputs)
        learn_rate=0.0005

    elif name=='16': # Resnet 50 
        model = ResNet50(img_input=inputs)
        learn_rate=0.00001

    elif name=='17': # Resnet 50 
        model = ResNet50(img_input=inputs)
        learn_rate=0.000005

    elif name=='18': # Resnet 18 
        model = ResNet18(img_input=inputs)
        learn_rate=0.0005

    elif name=='19': # Resnet 18 
        model = ResNet18(img_input=inputs)
        learn_rate=0.00001

    elif name=='20': # Resnet 18 
        model = ResNet18(img_input=inputs)
        learn_rate=0.000005

    elif name=='21': # Resnet 18 
        model = ResNet18(img_input=inputs)
        learn_rate=0.000001

    ## Add more models above
    ############################################
    ####### Compile model ######################
    ############################################

    if name in ['15','16','17','18','19','20','21'] :
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'sparse_categorical_crossentropy'

    else : ## For non resnet models 
        model = models.Model(inputs, outputs)
        learn_rate=0.0005
        #### change loss function for non-resnet models since 'sparse_categorical_crossentropy' throws up an error.
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'binary_crossentropy'
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    #print("model %s"%name)
    #model.summary()

    return model




