# HAR
Human Activity Recognition using Continuous Wavelet Transform and Convolutional Neural Networks


## Introduction

The problem in question is _human activity recognition based on accelerometer signal using continuous wavelet transform and convolutional neural networks_. All the code is split into five files: _main.ipynb_, _Creation_Crop.py_, _Models.py_, _Preprocessing_Training.py_ and _Graphics.py_. This files should be placed in the same folder with the dataset file _full_data.mat_ (from _UniMiB SHAR_ dataset). Files with code are described below.


## Creation_Crop.py

_Creation_Crop.py_ contains functions for WT application and primary image crop. The output files of these functions should be saved into separate folders.


## Preprocessing_Training.ipynb

_Preprocessing_Training.ipynb_ является центральным файлом работы. С его помощью происходит нарезка изображений заданных размеров с установленным шагом, разбиение полученной выборки на тренировочную и тестовую, построение свёрточной модели и её обучение. Описанные в нем функции можно импортировать из файла _Creation_Crop.py_

Ноутбук работает с изображениями, полученными в при помощи функции _cut_image_ предыдущего ноутбука. В результате выполнения ноутбука создаются файлы _metrics.csv_ (пользовательское задание имени файлов не предусмотренно, поэтому их необходимо переименовывать вручную).

## Graphics.ipynb

_Graphics.ipynb_ - вспомогательный ноутбук. В нём описана функция, позволяющая быстро и легко строить графики для значений метрик (необходимо подавать эти файлы программе в виде файлом расширения _.csv_, можно переименованных), посчитанных в предыдущем ноутбуке. Эта функция может быть импортирована из файла _Graphics.py_.

# Models.py

Из данного файла могут быть импортированы свёрточные модели, используемые в ходе выполнения программы.

# Function_implementation

Это основной файл программы. В нём происходит импорт всех необходимых функций и их применение для создания изображений, обучения моделей, построения графиков и т. д.
