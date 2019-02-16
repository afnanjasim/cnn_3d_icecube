## modify this script to change the model.
### Add models with a new index.

from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras

### Defining all the models tried in the study
def f_define_model(inpx,name):
    '''
    Function that defines the model and compiles it.
    '''
    
    inputs = layers.Input(shape=inpx.shape[1:])
    h = inputs
    
    # Choose model
    if name=='1':
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        print("model %s"%name)
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
        
 
    ############################################
    ####### Compile model ######################
    ############################################
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()

    return model

