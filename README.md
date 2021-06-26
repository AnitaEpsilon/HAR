# HAR
Human Activity Recognition using Continuous Wavelet Transform and Convolutional Neural Networks


## Introduction
The problem in question is _human activity recognition based on accelerometer signal using continuous wavelet transform and convolutional neural networks_. All the useful code is split into four files. _Function_implementation_ is, in fact, a brief instruction of how to use the program. All the files and the data file _full_data.mat_ should be placed in the same directory. All the files are described below.

## Creation_Crop.ipynb





_Creation_Crop.ipynb_ является первым в хронологической последовательности файлов. С его помощью можно генерировать изображения, представляющие собой результаты применения непрерывного вейвлет преобразования к исходным сигналам, и обрезать белую рамку вокруг этих изображений. Функции, описанные в файле _Creation_Crop.ipynb_ могут быть импортированы из файла _Creation_Crop.py_

Для запуска необходим файл _full_data.mat_.
Результат каждого вызова функций _apply_wavelet_transform_ и _cut_image_ необходимо хранить в отдельной директории.

## Preprocessing_Training.ipynb

_Preprocessing_Training.ipynb_ является центральным файлом работы. С его помощью происходит нарезка изображений заданных размеров с установленным шагом, разбиение полученной выборки на тренировочную и тестовую, построение свёрточной модели и её обучение. Описанные в нем функции можно импортировать из файла _Creation_Crop.py_

Ноутбук работает с изображениями, полученными в при помощи функции _cut_image_ предыдущего ноутбука. В результате выполнения ноутбука создаются файлы _metrics.csv_ (пользовательское задание имени файлов не предусмотренно, поэтому их необходимо переименовывать вручную).

## Graphics.ipynb

_Graphics.ipynb_ - вспомогательный ноутбук. В нём описана функция, позволяющая быстро и легко строить графики для значений метрик (необходимо подавать эти файлы программе в виде файлом расширения _.csv_, можно переименованных), посчитанных в предыдущем ноутбуке. Эта функция может быть импортирована из файла _Graphics.py_.

# Models.py

Из данного файла могут быть импортированы свёрточные модели, используемые в ходе выполнения программы.

# Function_implementation

Это основной файл программы. В нём происходит импорт всех необходимых функций и их применение для создания изображений, обучения моделей, построения графиков и т. д.
