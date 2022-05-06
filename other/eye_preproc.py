import numpy as np
from scipy import signal
from saclabel import tools

# load the data
x_matrix,y_matrix  = tools.read_eye_data_mat()

# get the dimensions of the data
ntrials        = np.shape(x_matrix)[0]
FS             = 1000 # sampling rate in Hz
freq_to_remove = [50,100,150]

for iTrial in range(ntrials):
    this_x = x_matrix[iTrial]
    this_y = y_matrix[iTrial]
    # flatten to find min and max
    flat_list_x = [item for sublist in x_matrix for item in sublist]
    flat_list_y = [item for sublist in y_matrix for item in sublist]

    for nharmonic in freq_to_remove: # remove the first n harmonics of line noise
        b, a = signal.iirnotch(nharmonic,10,FS)
        x_matrix[iTrial] = signal.filtfilt(b, a, x_matrix[iTrial])
        y_matrix[iTrial] = signal.filtfilt(b, a, y_matrix[iTrial])

        


