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

val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Dataset/train',
    target_size=(64,64),
    batch_size=10,
    class_mode='binary')

val_set = val_datagen.flow_from_directory(
    'Dataset/validation',
    target_size=(64,64),
    batch_size=10,
    class_mode='binary')

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
        epochs=100, 
        verbose=1,
        validation_data = val_set,
        validation_steps=266//10,
        callbacks=[callback]
    )

# %%
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#%%
path = 'kiwi.jpg'
img_original = load_img(path)
img = load_img(path, target_size = (64,64))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor/=255.0

# %%

pred = model.predict(img_tensor)
print(pred)

# %%
if pred<.5: str = '--------------Orange--------------'
else: str = '--------------Non Orange--------------'

# %%
plt.imshow(img_original)
plt.axis('off')
plt.title(str)
plt.show()

# %%
''' 
okay so far my model is baised. Since i just traing my model on the
images of the orange. It is classifing every image as orange.
need to add another class here stating that those are not oranges.

''' 