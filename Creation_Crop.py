from scipy.io import loadmat
import torch
import scipy.io as spio
import numpy as np
import pycwt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from tqdm import tqdm

from PIL import Image
from numpy import asarray


# Dataset loading and data preprocessing
mat = spio.loadmat('full_data.mat', squeeze_me=False)


# Function data_parser(mat_array) is designed to split the original dataset mat_array into two subsets: people_metadata (includes information about test subjects, namely their age, sex, height and weight) and people_data (includes acceleration data for different motion types)

def data_parser(mat_array):
    '''
    INPUT: initial numpy.array from spio.loadmat()
    OUTPUT: numpy.array 'people'
    values in element 'people_metadata[i][j]' contains information about:
    i - number of the subject
    j - type of metadata: j == 0: string:     gender
                               1: interger:   age
                               2: interger:   height
                               3: interger:   weight
    
    values in element 'signal_data[i][key]' contains information about:
    i - number of the subject
    key = type of activity:    'StandingUpFS'
                               'StandingUpFL'
                               'Walking'
                               'Running'
                               'GoingUpS'
                               'Jumping'
                               'GoingDownS'
                               'LyingDownFS'
                               'SittingDown'
                               'FallingForw'
                               'FallingRight'
                               'FallingBack'
                               'HittingObstacle'
                               'FallingWithPS'
                               'FallingBackSC'
                               'Syncope'
                               'FallingLeft'
    value = data from accelerometer, where k - number of the trial,
    value[k] contains information in format: (x,y,z, time instants, magnitudo) * number of timesteps                                                             
    '''
    
    preprocess = mat_array['full_data'].copy()
    number_of_people = len(preprocess)
    
    # reserve memory for output
    people_metadata = np.empty((number_of_people, 4), dtype=object)
    people_data = np.empty((number_of_people), dtype=object)

    dictionary = {}
    
    # fill arrays with data and metadata
    for number in np.arange(number_of_people):
        
        # metadata
        people_metadata[number, 0] = str(preprocess[number][1][0]).strip()
        people_metadata[number, 1:] = [int(preprocess[number][i][0]) for i in np.arange(2, 5)]
        
        # signals data
        # extract names of activity from preprocess
        signal_data = preprocess[number][0]
        dtype_ = signal_data.dtype
        list_of_activity = dtype_.names
        
        # fill dictionary with specific activities
        for names in list_of_activity:
            dictionary[names] = signal_data[names].flatten()[0].squeeze()
        
        people_data[number] = dictionary
        dictionary = {}
        
    return people_metadata, people_data


# Apply function data_parser to the original data:
metadata, data = data_parser(mat)


# There are 17 types of action in the dataset including 9 types of ADLs and 8 types of Falls:

activities = [
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

# In the following section WT-parameters are set

t0 = 0  # Initial time value
dt = 1  # Time step
s0 = 2 * dt  # Initial scale value

dj = 1 / 12  #Scale step
J = 7 / dj  # Number of different scale values

number_of_people = data.size



# Function apply_wavelet_transform realization

# Function wavelet_function allows to apply CWT to dataset elements
# Arguments:
# 
# * coordinate - coordinate axis (0 - X, 1 - Y, 2 - Z, 5 - magnitude),
# * output_path - output director pathway
# * mother - mother wavelet with respact to the zero moment order (for example, mother = pycwt.Paul(m=3))

def apply_wavelet_transform(coordinate, output_path, mother):

    for i in tqdm(range(number_of_people)):
        for activity in activities:
            for trial in range(len(data[i][activity])):
  
                current_series = data[i][activity][trial][coordinate]

                N = current_series.size
                t = np.arange(0, N)

                p = np.polyfit(t - t0, current_series, 1)
                dat_notrend = current_series - np.polyval(p, t - t0)
                std = dat_notrend.std()  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend / std  # Normalized dataset

                wave, scales, freqs, _, fft, _ = pycwt.cwt(dat_norm, dt, dj, s0, J, mother)

                power = (np.abs(wave)) ** 2
                power /= scales[:, None]
                fft_power = np.abs(fft) ** 2
                period = 1 / freqs

                power = (np.abs(wave)) ** 2
                power /= scales[:, None]
                fft_power = np.abs(fft) ** 2
                period = 1 / freqs

                f, axs = plt.subplots(1, 1, figsize=(len(t)/100.0, 0.7))
                levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
                plt.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                             extend='both', cmap=plt.cm.viridis)

                plt.savefig(output_path + str(activity) + '_' + str(i) + '_tr' + str(trial) + '.png')   


# Function cut_image realization

# Function cut_image crops white pixels around images. Default parameters correspond to standart image size. If the used images have another size the pozitions of white pixels can be determined via any graphical redactor
# Arguments:
# 
# * input_path - input directory pathway,
# * output_path - output directory pathway,
# * left, right, top, bottom - primary cropping coordinates
# * left_new, right_new, top_new, bottom_new - secondary cropping coordinates



def cut_image(input_path, output_path, 
              left=None, right=None, top=6, bottom=44,
              left_new=1, right_new=2, top_new=1, bottom_new=37):
    
    for filename in os.listdir(input_path):

        current_image = Image.open(input_path+filename)
        width, height = current_image.size
        data = asarray(current_image)
        left = 0
        right = left + width
        transp = np.transpose(data, (2, 1, 0))[0]    # Take a color channel
        image = current_image.crop((left, top, right, bottom)) 
        
        data = asarray(image)
        transp = np.transpose(data, (2, 1, 0))[0]
        for i in range(len(transp)):
            if transp[i][18] != 255:    # Search for the last non-white pixel
                left_new = i+1
                break
                
        for i in range(len(transp)):
            if transp[len(transp)-i-1][18] != 255:   # Search for the first non-white pixel
                right_new = len(transp)-i-1
                break

                
        image = image.crop((left_new, top_new, right_new, bottom_new))     # Image crop
        image.save(output_path + str(filename))

