#%%
try:
    import sys,os
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    from build_model import classifier_model
    print("----Libraries Loaded----")
except:
    print("----Libraries Not Loaded----")

#%%
os.listdir()

#%%
tf.debugging.set_log_device_placement(True)

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1. / 255, 
        rotation_range=30,  
        zoom_range = 0.15,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Dataset/train',
    target_size=(64,64),
    batch_size=15,
    class_mode='input')

test_set = test_datagen.flow_from_directory(
    'Dataset/test',
    target_size=(64,64),
    batch_size=15,
    class_mode='input')

# %%
model = classifier_model()

model.summary()

# %%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.95):
            print("Reached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callback = myCallback()

# %%
with tf.device('/GPU:0'):
    history = model.fit(
        training_set,
        steps_per_epoch=591//10,
        batch_size=32, 
        epochs=100, 
        verbose=1,
        validation_data=test_set,
        validation_steps=266//10,
        callbacks=[callback]
    )
# %%
