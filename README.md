# Orange-Fruit-Recognition-Using-Image-Segmentation
> Identifying whether input image contains Orange or Not using Image Segmentaion Technique.

## Aim of the Project
1. To identify Orange or Not. (Classification)
2. Creation Of Bounding Box around the classified object(orange). (Classification  +  Localisation)
3. Using that Localized portion for Segmenting Orange. (Image Segmentation)


Note: Kindly do not push any changes to Main or Master Branch. Create a New Branch and push all the changes to that branch.

Don't forget to create an issue before making a PR.

👉 Repo Link : https://github.com/Ayush12062000/Orange-Fruit-Recognition-Using-Image-Segmentation.git

## Table of contents
* About Project
* Dataset
* Languages or Frameworks Used
* Setup

## About Project
For the recognition of Orange fruit, Input image of the orange which is captured at different lighting conditions will be recognised
and Orange from that image will be segmented using different colors.

## Dataset
The Dataset used is self created. You can download the dataset from here - 
[Orange_Fruit_Dataset](https://github.com/Ayush12062000/Orange-Fruit-Recognition-Using-Image-Segmentation/tree/main/Dataset) , It
contains 1062 images and 526 images in train and validation folders respectively. There are 2 classes/categories in this
dataset (0=Not Orange, 1=Orange). 

## Languages or Frameworks Used
* Python: language
* NumPy: library for numerical calculations
* Pandas: library for data manipulation and analysis
* Tensorflow: library for large numerical computations without keeping deep learning in mind
* Keras: neural network library
* Matplotlib: for creating static, animated, and interactive visualizations

## Setup

1. First Clone the repository.
2. If you are working on your local Machine then:-
	* Create and activate the virtual environment for the project.
		```	
		$ python -m venv Project_emotion
		$ Project_emotion\Scripts\activate.bat
		```
	* Install the required packages using requirements.txt inside the environemnt using pip.
		```
		$ pip install -r requirements.txt
		```
	* Code Orange_Fruit_Part1.ipynb and build_model.py in your local machine.
3. If you want to work on Google Colab then:-
	* Upload Dataset, build_model.py, Orange_Fruit_Part1.ipynb, and Images Folder in you drive.
	* Run the Orange_Fruit_Part1.ipynb in google colab (make sure GPU/TPU is enabled).

