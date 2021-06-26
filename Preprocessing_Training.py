from __future__ import print_function
import torch
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.io import loadmat
import os
from tqdm import tqdm
import copy

import pandas as pd
import matplotlib.pyplot as plt
import os

get_ipython().run_line_magic('tensorflow_version', '1.x')
from torchvision.transforms import Compose, Grayscale
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_grayscale

from PIL import Image, ImageDraw 
from sklearn.model_selection import train_test_split, cross_val_score

import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import KFold

from numpy.random import seed
from numpy import asarray
from tensorflow import set_random_seed

import tensorflow as tf
import datetime
import glob


keras.applications.resnet_v2.ResNet50V2()
keras.applications.resnet_v2.ResNet101V2()
keras.applications.resnet_v2.ResNet152V2()

keras.applications.resnet.ResNet50()
keras.applications.resnet.ResNet101()
keras.applications.resnet.ResNet152()


# Preprocessings
# List classes  consists of 17 types of actions, including 9 ADLs and 8 Falls:

classes = [
    'StandingUpFS',
    'StandingUpFL',
    'Walking',
    'Running',
    'GoingUpS',
    'Jumping',
    'GoingDownS',
    'LyingDownFS',
    'SittingDown',
    'FallingForw',
    'FallingRight',
    'FallingBack',
    'HittingObstacle',
    'FallingWithPS',
    'FallingBackSC',
    'Syncope',
    'FallingLeft'
]


# Method transforms.Compose combines methods transforms.ToTensor 
# (converts image with pixel diapason [0, 255] into tensor with [0, 1] diapason) 
# and transforms.Normalize (normalizes images) into a method transform:

transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


# Dictionary dict_classes assigns a unique index to every class: 

dict_classes = {}
i = 0
for activity in classes:
    dict_classes[activity] = i
    i += 1


# Function preprocess_function processes иoriginal images and forms a benchmarked dataset. 
# Arguments:
# 
# * input_path - input directory pathway,
# * sample_width - width of cropped images,
# * step - step of frame shift,
# * list_of_folders - list of directories with original images,
#
# Returns:
# 
# * array - tensor of cropped and preprocessed images,
# * targets - vetor of labels


def preprocess_function(input_path, 
                        multiframe=True,
                        sample_width=90, step=15,
                        list_of_folders=[]):
    
    
    array = []                             # List of final images 
    targets_words = []                     # Vector of labels (consists of action names, for example Running)
    targets = []                           # Encoded vector of labels   
    
    
    
    """Here we test if there is at least one directory in the list.
       If not, the function stops running and returns empty arrays"""
    
    if len(list_of_folders)==0:
        print("No folder is chosen. Chose at least one folder, please")
        return [], []    
    
    

    path = os.path.join(input_path, list_of_folders[0])
    
    
    
    """Here we test if all directories have the same number of images.
       If not, the function stops running and returns empty arrays"""

    def listdir_nohidden(path):
        return glob.glob(os.path.join(path, '*'))      # ignore hidden files (including '.ipynb_checkpoints')

    if len(list_of_folders) > 1:
        for i in range(len(list_of_folders)-1):
            path_1 = os.path.join(input_path, list_of_folders[i])
            path_2 = os.path.join(input_path, list_of_folders[i+1])
            if len(listdir_nohidden(path_1)) != len(listdir_nohidden(path_2)):
                print("Error!  There are different number of files in folders!")
                return [], []
    
    
    
    """Search for the minimal image size min_width."""
 
    all_widths = []
    
    for filename in tqdm(os.listdir(path)):
        if filename==".ipynb_checkpoints":
            continue 
        current_image = Image.open(os.path.join(input_path, list_of_folders[0], filename))
        data = asarray(current_image)
        height, width, dimensions = data.shape
        all_widths.append(width)
        
    min_width = min(all_widths)
        
        
    """Here we iterate on images from the chosen directory.
       If any images is not in every directory, 
       the function stops running and returns empty arrays.
       It is assumed that images in different directories 
       related to the same object are identical in size"""    
    
    for filename in tqdm(os.listdir(path)):

        if filename==".ipynb_checkpoints":
            continue
        
        try:
            data_array = []
            for folder in list_of_folders:
                current_image = Image.open(os.path.join(input_path, folder, filename))               
                data_array.append(current_image)
            data = asarray(current_image)
            height, width, dimensions = data.shape
            
        except:
            print("Image " + str(filename) + " has not all the chosen axes")
            return [], []
        
        
        if multiframe:          
            for j in range(int(round((width - sample_width)/step + 1))):

                # Image size
                left = step*j
                right = left + sample_width
                top = 0
                bottom = copy.copy(height)

                # Images along all chosen axes
                images = []

                for data in data_array:
                    im = data.crop((left, top, right, bottom)) 
                    im = transforms.functional.to_grayscale(im, 1)
                    im = np.array(im)
                    images.append(im)                  

                images = np.array(transform(np.array(images)))
                array.append(images)
                targets_words.append(filename)  
                
        else: 
            
            # Image size
            left = 0
            right = copy.copy(min_width)
            top = 0
            bottom = copy.copy(height) 
            
            # Images along all chosen axes
            images = []
            
            for data in data_array:
                im = data.crop((left, top, right, bottom)) 
                im = transforms.functional.to_grayscale(im, 1)
                im = np.array(im)
                images.append(im)
                
            images = np.array(transform(np.array(images)))
            array.append(images)
            targets_words.append(filename) 
                
                            
                  
    """Now we have 2 lists: arrays (contains tensors of cropped 
       and preprocessed images) and targets_words (vector of labels). 
       Let's encode categorical targets_words to targets"""

    for i in range(len(targets_words)):
        targets_words[i] = targets_words[i][:-10]
        if targets_words[i][-1] == '_':
            targets_words[i] = targets_words[i][:-1]

    for word in targets_words:
        targets.append(dict_classes[word])
    targets = np.array(targets)

    targets = to_categorical(targets, num_classes=len(classes))
    targets = np.array(targets)

    return array, targets    


# Dataset split

# Function split_function splits dataset into train and test samples. 
# Arguments:
# 
# * array - original array of features,
# * targets - vector of labels,
# * test_size - test sample fraction (0.2 by default),
# * random_state - random state value (26 by defaut).
# 
# Function split_function returns train and test samples and corresponing vector of labels:
# 
# * X_train, 
# * X_test, 
# * y_train, 
# * y_test.


def split_function(array, targets, test_size=0.20, random_state=26):
    
    X_train, X_test, y_train, y_test = train_test_split(array, targets, 
                                                        test_size=test_size, 
                                                        shuffle=True, 
                                                        random_state=random_state)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    
    return X_train, X_test, y_train, y_test


# Now we build neural network

# my_callback is a daughter class of the class callbacks:

class my_callback(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super().__init__()
        self.acc = []
        self.loss = []
        self.rec = []
        self.prec = []     
 

    """After every epoch metrics are counted via cross-validation"""
    
    def on_epoch_end(self, epoch, logs=None):
        l, a, r, p = self.model.evaluate(self.validation_data[0], 
                                         self.validation_data[1])
        self.loss.append(l)
        self.acc.append(a)
        self.rec.append(r)
        self.prec.append(p)

        
    """After training is complete metrics are saved into metrics.csv"""    
    
    def on_train_end(self, logs=None):

            df = {'loss': self.loss,
                  'accuracy': self.acc,
                  'recall': self.rec,
                  'precision': self.prec}
            df = pd.DataFrame(data=df)
            df.to_csv("metrics.csv", index=False) 


# Function create_and_run_model builds and trains a neural network and creates a file metrics.csv containing metrics.
# Arguments:
# 
# * X_train, X_test, y_train, y_test - train and test samples and corresponding vectors of labels,
# * res_model - ResNet model from Keras (keras.applications.resnet_v2.ResNet50V2, 
#                                        keras.applications.resnet_v2.ResNet101V2, 
#                                        keras.applications.resnet_v2.ResNet152V2,
#                                        keras.applications.resnet.ResNet50,
#                                        keras.applications.resnet.ResNet101, 
#                                        keras.applications.resnet.ResNet152),
# * number_of_classes - number of different types of activity, 
# * height - input image height,
# * width - input image width,
# * dimensions - number of channels of input images,
# * batch_size - batch size,
# * epochs - number of epochs.


def create_and_run_model(X_train, X_test, y_train, y_test,
                         resnet = True,
                         base_model=keras.applications.resnet_v2.ResNet50V2,
                         number_of_classes=len(classes),
                         height=36,
                         width=90,
                         dimensions=3,
                         batch_size=200,
                         epochs=300):

    if resnet:
        model = base_model(weights=None, input_shape=(height, width, dimensions),
                         classes=number_of_classes)
    else:
        model = base_model
    	                    

    model.compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[my_callback()])
