# Orange-Fruit-Recognition-Using-Image-Segmentation
> Identifying whether input image contains Orange or Not using Image Segmentaion Technique.

## Aim of the Project
1. To identify Orange or Not. (Classification)
2. Using that Classified image for Segmenting Orange using UNet. (Image Segmentation)


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
The Dataset used is self created. And two different datasets having 1588 and 295 samples of orange
fruit and fruits other than orange are used. The first dataset contains 794 samples of orange
fruit images and 794 images of fruits that are other than orange. Multiple fruits or vegetables
apart from orange fruit images were used that is Apple, Banana, Guava and Pepper Orange,
broccoli, Onion, Bitter gourd, etc. In the second dataset, a total set of 295 images are used and
this dataset also contains images of oranges and there corresponding masks.

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
	* Code Orange_Fruit_Part1.ipynb and build_model.py in your local machine and save the model.
	* Then, use that classifier to classify orange and send that image as input to UNet Model.
	* code main.py and UNet_Model.py to get segmented orange.
3. If you want to work on Google Colab then:-
	* Upload Dataset, train_data, test_data, build_model.py, Orange_Fruit_Part1.ipynb, main.py, UNet_Model.py and Images Folder in you drive.
	* Run the Orange_Fruit_Part1.ipynb, and main.py in google colab (make sure GPU/TPU is enabled).

