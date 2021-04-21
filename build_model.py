#%%
import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy  
from keras.optimizers import Adam  
from keras.regularizers import l2 
from keras.utils import np_utils


#%%
def classifier_model():
    
    # initializing the sequential model
    model = Sequential()

    # adding first bolck of the model
    
    model.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu' , kernel_initializer='random_normal'))
    model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # adding second block of the model
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # adding third layer of the model
    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'random_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # Flattening the model
    model.add(Flatten())

    # adding the Dense layers
    model.add(Dense(256, activation='relu' , kernel_initializer='random_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu' , kernel_initializer='random_normal'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # adding the output layer
    model.add(Dense(1, activation='sigmoid' , kernel_initializer='random_normal'))

    # model compilation + optimizer and loss function specifications.
    # opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model

# %%
