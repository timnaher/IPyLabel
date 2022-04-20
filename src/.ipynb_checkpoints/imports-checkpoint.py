from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from csv import writer
import os.path
import pandas as pd


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

