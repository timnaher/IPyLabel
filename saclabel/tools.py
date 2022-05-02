import os.path
from csv import writer

import mat73
import numpy as np
import pandas as pd
import uneye
from numpy import genfromtxt


def read_eye_data_mat():
    """ this function reads the eye data from -v7.3 mat files

    Returns:
        _type_: _description_
    """    
    eye_x = mat73.loadmat('data/eyedata_x.mat')['eyedata_x']
    eye_y = mat73.loadmat('data/eyedata_y.mat')['eyedata_y']
    return eye_x, eye_y


def read_eye_data():
    """ reads the eye date from the csv file

    Returns:
        x_data, y_data: 2 numpy arrays with eye x and y traces
    """
    # import X and Y eye traces
    x_data = genfromtxt('data/X_train.csv', delimiter=',') # dimensions are trials x time
    y_data = genfromtxt('data/Y_train.csv', delimiter=',')
    return x_data, y_data


def append_list_as_row(file_name, list_of_elem):
    """ appends to an existinc csv file a new row of elements

    Args:
        file_name (string): the file name on which to append
        list_of_elem (list): the elements which to append as a new row
    """
    with open(file_name, 'a+', encoding="utf-8", newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def check_results_file():
    """ checks if results file is there and if not, creates empty file with a row
    of zeros of length length_zero_list

    Args:
        length_zero_list (int): length of the empty list that is first row
    """
    if not os.path.isfile('./results/binary_labels.csv'):
        df = pd.DataFrame(list())
        df.to_csv('./results/binary_labels.csv')
    if not os.path.isfile('./results/eye_x.csv'):
        df = pd.DataFrame(list())
        df.to_csv('./results/eye_x.csv')
    if not os.path.isfile('./results/eye_y.csv'):
        df = pd.DataFrame(list())
        df.to_csv('./results/eye_y.csv')


def get_sac_onsets_offsets(total_predictions):
    '''gets the onset and offset of  change based on the binary prediction vector'''
    ntrials = np.shape(total_predictions)

    onsets  = []
    offsets = []
    for j in range(ntrials):
        onsets.append( np.asarray( np.where( np.diff( total_predictions[j,:] ) ==  1)) + 1)
        offsets.append(np.asarray( np.where( np.diff( total_predictions[j,:] ) == -1))    )
    return onsets, offsets



def predict_possible_saccades(x,y,sampfreq,min_sacc_dur,min_sacc_dist):
    """ predicts saccades based on general weights from other datasets

    Args:
        x (_type_): x eye trace
        y (_type_): y eye trace
        sampfreq (int: the Fs in Hz
        min_sacc_dur (int): the minimum saccade duration for the network to detect
        min_sacc_dist (int): the minimum distance between 2 saccades, otherwise merge

    Returns:
        prediction (np.ndarray): binary prediction of saccades (1) or fixation (0)
    """
    sampfreq = 1000 #Hz
    weights_name = 'weights_1+2+3'
    min_sacc_dur = 6 # in ms
    min_sacc_dist = 10 #in ms

    x,y = read_eye_data()

    model = uneye.DNN(weights_name=weights_name,
                    sampfreq=sampfreq,
                    min_sacc_dur=min_sacc_dur,
                    min_sacc_dist=min_sacc_dist)

    prediction = model.predict(x,y)
    return prediction
