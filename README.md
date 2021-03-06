# HAR
Human Activity Recognition using Continuous Wavelet Transform and Convolutional Neural Networks


## Introduction

The problem in question is _human activity recognition based on accelerometer signal using continuous wavelet transform and convolutional neural networks_. All the code is split into five files: _main.ipynb_, _Creation_Crop.py_, _Models.py_, _Preprocessing_Training.py_ and _Graphics.py_. These files should be placed in the same folder with the dataset file _full_data.mat_ (from _UniMiB SHAR_ dataset). Files with code are described below.


## main.ipynb

_main.ipynb_ demonstrates some of the experiments described in the paper. 


## Creation_Crop.py

_Creation_Crop.py_ contains functions for WT application and primary image crop. The output files of these functions should be saved into separate folders.


## Preprocessing_Training.py

_Preprocessing_Training.py_ contains functions used for secondary image crop (with a shift or without it) and model training. The functions from these file are desisigned to deal with images obtained via using the _cut_image_ from the previous notebook. In result of fuction implementation files _metrics.csv_ are created. User-defined file naming is not provided, so user has to rename them manually.


## Graphics.py

_Graphics.ipynb_ contains a function *create_plot* designed to plot metrics using files _metrics.csv_ or their renamed versions.


## Models.py

_Models.py_ contains functions which allow to create various convolutional neural networks which are used in the course of the studying.
