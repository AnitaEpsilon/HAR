import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import os
import numpy as np


# Function _create_plot_ realization:

# Function _create_plot_ plots metrics _loss_, _accuracy_, _recall_ и _precision_.
# Arguments:
# 
# * _path_ - input directory pathway ,
# * _list_of_files_ - list of files with metric values,
# * _locs_ - legend pozitions list,
# * _colors_ - list of colors.



def create_plot(path, list_of_files, list_of_names, 
                locs=['upper right', 'lower right', 'lower right', 'lower right'],
                colors = ['red', 'gray', 'black', 'maroon', 'royalblue',
                          'olive', 'darkcyan', 'lightcoral']):
    
    plt.style.use('default')
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    
    
    i = 0
    for f in list_of_files:
        
        color = colors[i]
        metrics = pd.read_csv(os.path.join(path, f))
        
        type(metrics)
        loss = list(metrics.loss)
        accuracy = list(metrics.accuracy)
        recall = list(metrics.recall)
        precision = list(metrics.precision)
        
        axs[0][0].plot(np.arange(len(loss)), loss, color=color, label=list_of_names[i])
        axs[1][0].plot(np.arange(len(accuracy)), accuracy, color=color, label=list_of_names[i])
        axs[0][1].plot(np.arange(len(recall)), recall, color=color, label=list_of_names[i])
        axs[1][1].plot(np.arange(len(precision)), precision, color=color, label=list_of_names[i])
        
        i += 1

    axs[0][0].set(xlabel='epochs')
    axs[1][0].set(xlabel='epochs')
    axs[0][1].set(xlabel='epochs')
    axs[1][1].set(xlabel='epochs')
    
    axs[0][0].set(ylabel='loss function')
    axs[0][1].set(ylabel='recall')
    axs[1][0].set(ylabel='accuracy')
    axs[1][1].set(ylabel='precision')

    axs[0][0].legend(loc=locs[0])
    axs[0][1].legend(loc=locs[1])
    axs[1][0].legend(loc=locs[2])
    axs[1][1].legend(loc=locs[3])
    plt.savefig('acc_loss_rec_prec')
    plt.show()
