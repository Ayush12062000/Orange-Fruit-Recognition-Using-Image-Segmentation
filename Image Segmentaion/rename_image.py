#%%
import os

#%%
# for train data
os.chdir('E:/Projects 6th SEM/Orange-Fruit-Recognition-Using-Image-Segmentation/Image Segmentaion/train_data')

# for test data
# os.chdir('E:/Projects 6th SEM/Orange Fruit Resources/test_data')


# %%
os.listdir()

# %%

# train data
for i, filename in enumerate(os.listdir()):
    x = os.path.join(filename+'/images')
    for j in enumerate(os.listdir(x)):
        #print(j[1])
        
        os.rename(os.path.join(x,j[1]), os.path.join(x,filename + '.jpg'))
    
    
    
#%%

# test data
for i, filename in enumerate(os.listdir()):
    x = os.path.join(filename+'/images')
    for j in enumerate(os.listdir(x)):
        #print(j[1])
        
        os.rename(os.path.join(x,j[1]), os.path.join(x,filename + '.jpg'))
        
# %%
