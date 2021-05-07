#%%
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

#%%
# Loading Model

def Loading_Model():
    model = model_from_json(open("Classification_model.json", "r").read())  
    model.load_weights('Orange_Fruit_Weights2.h5')
    return model

#%%
# Loading and Processing image
def predicting(path,model):
    
    def process_image(path):
        img = load_img(path, target_size = (64,64))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis = 0)
        img_tensor/=255.0
        return img_tensor
    img_original1 = load_img(path)
    pred = model.predict(process_image(path))
    def show(pred,img):
        if pred>0.5: 
            str = '--------------Orange--------------'
        else:
            str = '--------------Not Orange--------------'
        plt.imshow(img)
        plt.axis('off')
        plt.title(str)
        plt.show()
    
    show(pred,img_original1)
    
    return pred
