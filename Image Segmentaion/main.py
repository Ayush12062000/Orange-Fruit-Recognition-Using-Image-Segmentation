#%%
# Import Required Libraries
try:
    import tensorflow as tf
    import os
    import random
    import numpy as np
    from tqdm import tqdm 
    from skimage.io import imread, imshow
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from keras.models import model_from_json
    print("----Libraries Imported----")
except:
    print("----Libraries Not Imported----")


#%%
# checking the content of the current directory
os.listdir()

#%%
# Setting up path
seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


TRAIN_PATH = 'E:/Projects 6th SEM/Orange-Fruit-Recognition-Using-Image-Segmentation/Image Segmentaion/train_data/'
TEST_PATH = 'E:/Projects 6th SEM/Orange-Fruit-Recognition-Using-Image-Segmentation/Image Segmentaion/test_data/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

print(train_ids)
print(test_ids)

#%%
# Loading data

# independent variable
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# dependent variable (what we are trying to predict)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) 

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.jpg')[:,:,:IMG_CHANNELS] 
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.jpg')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

#%%
# Showing Random images from the dataset

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()


#%%
from UNet_Model import Segmentation_model

model = Segmentation_model()
model.summary()

#%%

################################
#Modelcheckpoint

with tf.device('/GPU:0'):
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=100)


print('Training DONE')

#%%
# Plotting Training Results

plt.plot(results.history['accuracy'][0:150])
plt.plot(results.history['val_accuracy'][0:150])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_accuracy', 'validation_accuracy'])
plt.show()


#%%

plt.plot(results.history['loss'][0:150])
plt.plot(results.history['val_loss'][0:150])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'])
plt.show()


#%%
# Saving model

orange_model_json = model.to_json()  
with open("Segmentation_model.json", "w") as json_file:  
    json_file.write(orange_model_json)  
model.save_weights("Orange_Fruit_Weights_segmentation.h5")


#%%
# Loading Classification Model

import Prediction_file as pf
classification_model = pf.Loading_Model()

#%%
# Prediction

path1 = 'Images/kiwi.jpg'
path2 = 'Images/Orange.jpg'

pred1 = pf.predicting(path1,classification_model)
pred2 = pf.predicting(path2,classification_model)


#%%
# Loading Unet
segmentation_model = model_from_json(open("Segmentation_model.json", "r").read())  
segmentation_model.load_weights('Orange_Fruit_Weights_segmentation.h5')

#%%
####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


#%%
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# %%
