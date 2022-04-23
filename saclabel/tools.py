import numpy as np
import matplotlib.pyplot as plt
from csv import writer
import os.path
import pandas as pd
import mat73
import uneye
from numpy import genfromtxt


def read_eye_data_mat():
    eye_x = mat73.loadmat('data/eyedata_x.mat')['eyedata_x']
    eye_y = mat73.loadmat('data/eyedata_y.mat')['eyedata_y']
    return eye_x, eye_y


def read_eye_data():
    # import X and Y eye traces
    X_data = genfromtxt('data/X_train.csv', delimiter=',') # dimensions are trials x time (in ms with Fs = 1e3)
    Y_data = genfromtxt('data/Y_train.csv', delimiter=',')
    return X_data, Y_data


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def check_results_file():
    # check if file is already present if not, make one
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
    ntrials,time = np.shape(total_predictions)

    onsets  = []
    offsets = []
    for j in range(ntrials):
        onsets.append( np.asarray( np.where( np.diff( total_predictions[j,:] ) ==  1)) + 1)
        offsets.append(np.asarray( np.where( np.diff( total_predictions[j,:] ) == -1))    )
    
    return onsets, offsets



def predict_possible_saccades(x,y,sampfreq,min_sacc_dur,min_sacc_dist):
    # Prediction
    sampfreq = 1000 #Hz
    weights_name = 'weights_1+2+3'
    min_sacc_dur = 6 # in ms
    min_sacc_dist = 10 #in ms

    x,y = read_eye_data()

    model = uneye.DNN(weights_name=weights_name,
                    sampfreq=sampfreq,
                    min_sacc_dur=min_sacc_dur,
                    min_sacc_dist=min_sacc_dist)

    Prediction,Probability = model.predict(x,y)

    return Prediction