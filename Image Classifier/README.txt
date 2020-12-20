There are some scripts that I do not recommend run, including Data_Info.py, Auto Crop.py and Manual Crop.py
Data_Info.py can generate a json file that could help further processing and face_data.json is for the face images
Auto Crop.py and manual Crop.py are for face recognition and face crop and some of codes are referred to: https://github.com/timesler/facenet-pytorch

# Loader.py is used to perform stratified cross validation on the dataset (note that there are two images with the same name and I just changed one of them manually)

# Evaluation.py contains module for model inference and visualization

# If you would like to try the code, just run Demo.py and change the dataset accordingly

# If it is to test the whole process, just run Classifier.py and change the dataset accordingly

# There is Processed directory containing all the cropped face images and Wrong directory contains all the images that require manual cropping

# Inference and Evaluation.ipynb is for model inference and confusion matrix, prior to that you need to get a trained .pth file by Demo.py or Classifier.py

# Example.ipynb is for demonstration of some mislabelled data from the database

# Requirements:
    CUDA 9.1.85
    Driver Version: 390.116
	Torch 1.0.0
	Torchvision 0.4.0
	Facenet-Pytorch 2.2.9
