# Code for Creating folders for dataset management

#%%
# importing os module
import os

# Directory
directory = "xyz"

# Parent Directory path
parent_dir = "E:/Projects 6th SEM/Orange-Fruit-Recognition-Using-Image-Segmentation/Image Segmentaion/test_data/"


for i in range(1,78):
    path = os.path.join(parent_dir, directory+str(i))
    
    os.mkdir(path)
    f1 = "images"
    path1 = os.path.join(path,f1)
    os.mkdir(path1)
    print("Directory '% s' created" % directory)


# %%
