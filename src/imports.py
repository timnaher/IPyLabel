from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

def read_eye_data():
    # import X and Y eye traces
    X_data = genfromtxt('data/X_train.csv', delimiter=',') # dimensions are trials x time (in ms with Fs = 1e3)
    Y_data = genfromtxt('data/Y_train.csv', delimiter=',')
    return X_data, Y_data

