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

def initial_model(input_shape):
  initial_model = models.Sequential()
  initial_model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape , kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  initial_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  initial_model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  initial_model.add(MaxPooling2D(pool_size=(2, 2)))
  initial_model.add(Flatten())
  initial_model.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  initial_model.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return initial_model


def model_3conv_layers(input_shape):
  model_3conv_layers = models.Sequential()
  model_3conv_layers.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape , kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_3conv_layers.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_3conv_layers.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  #####################
  model_3conv_layers.add(MaxPooling2D(pool_size=(2, 2)))
  model_3conv_layers.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_3conv_layers.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_3conv_layers.add(Flatten())
  model_3conv_layers.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  model_3conv_layers.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_3conv_layers


def model_32_128_128(input_shape):
  model_32_128_128 = models.Sequential()
  model_32_128_128.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape , kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_32_128_128.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_32_128_128.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_128_128.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_128_128.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_128_128.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_128_128.add(Flatten())
  model_32_128_128.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  model_32_128_128.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_32_128_128


def model_32_256_256(input_shape):
  model_32_256_256 = models.Sequential()
  model_32_256_256.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape, kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_32_256_256.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_32_256_256.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_256_256.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_256_256.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_256_256.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_256_256.add(Flatten())
  model_32_256_256.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  model_32_256_256.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_32_256_256


def model_32_192_192(input_shape):
  model_32_192_192 = models.Sequential()
  model_32_192_192.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape, kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_32_192_192.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_32_192_192.add(Conv2D(192, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_192_192.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_192_192.add(Conv2D(192, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_32_192_192.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_32_192_192.add(Flatten())
  model_32_192_192.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  model_32_192_192.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_32_192_192


def model_64_128_128(input_shape):
  model_64_128_128 = models.Sequential()
  model_64_128_128.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape, kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_64_128_128.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_64_128_128.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_64_128_128.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_64_128_128.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_64_128_128.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_64_128_128.add(Flatten())
  model_64_128_128.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  model_64_128_128.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_64_128_128


def model_2_extra(input_shape):
  model_2_extra = models.Sequential()
  model_2_extra.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                   activation='relu',
                   input_shape=input_shape, kernel_initializer='random_uniform',
                   bias_initializer='zeros'))
  model_2_extra.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model_2_extra.add(Conv2D(192, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_2_extra.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_2_extra.add(Conv2D(192, (5, 5), activation='relu', kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
  model_2_extra.add(MaxPooling2D(pool_size=(2, 2)))
  #####################
  model_2_extra.add(Flatten())
  model_2_extra.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  #####################
  model_2_extra.add(Dense(1000, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='relu'))
  #####################
  model_2_extra.add(Dense(output_dim=17, kernel_initializer='random_uniform',
                  bias_initializer='zeros', activation='softmax'))
  return model_2_extra