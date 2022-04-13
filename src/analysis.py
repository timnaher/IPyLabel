import uneye
import numpy as np
from src.imports import read_eye_data

def predict_possible_saccades(x,y,sampfreq,min_sacc_dur,min_sacc_dist):
    # Prediction
    sampfreq = 1000 #Hz
    weights_name = 'weights_synthetic'
    min_sacc_dur = 6 # in ms
    min_sacc_dist = 10 #in ms

    x,y = read_eye_data()

    model = uneye.DNN(weights_name=weights_name,
                    sampfreq=sampfreq,
                    min_sacc_dur=min_sacc_dur,
                    min_sacc_dist=min_sacc_dist)

    Prediction,Probability = model.predict(x,y)

    return Prediction





