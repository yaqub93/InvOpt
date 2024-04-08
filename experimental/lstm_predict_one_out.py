import sys
sys.path.append('../../src/preproc')
sys.path.append('../../models')

from math import sin, cos, sqrt, atan2, radians
from keras.models import model_from_json
from keras import optimizers
from keras import backend
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import r2_score
import folium


# User Inputs
GET_LATEST_FILE = False      # if true will get the the latest model from the models folder
model_name_json = 'dataset/models/samso_model_linear_60ep_2024-04-08-14-01.json'        # if false, specify filename here
model_name_h5 = 'dataset/models/samso_model_linear_60ep_2024-04-08-14-01.h5'          # if false, specify filename here

import pickle

# Open the pickle file in binary read mode ('rb')
with open('dataset/filtered_samso_lstm_18_ts_one_out.pkl', 'rb') as f:
    # Load data from the pickle file
    loaded_data = pickle.load(f)
x_test = loaded_data['X_test']
y_test = loaded_data['Y_test']

# Finding the most recent files
list_of_jsons = glob.glob('dataset/models/*.json')
list_of_jsons.sort(key=os.path.getctime, reverse=True)

list_of_h5 = glob.glob('dataset/models/*.h5')
list_of_h5.sort(key=os.path.getctime, reverse=True)

# Load json and create model
for i in range(len(list_of_jsons)):
    if GET_LATEST_FILE and 'samso' in str(list_of_jsons[i]):
        json_file = open(list_of_jsons[i], 'r')
        break
else:
    json_file = open(model_name_json, 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
for i in range(len(list_of_h5)):
    if GET_LATEST_FILE and 'samso' in str(list_of_h5[i]):
        loaded_model.load_weights(list_of_h5[i])
        print("Loaded most recent model from disk: \n", str(list_of_h5[i]))
        break
else:
    loaded_model.load_weights(model_name_h5)
    print("Loaded model based on user input: \n", model_name_h5)


# Function Definitions
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# Get distance between pairs of lat-lon points (in meters)
def distance(lat1, lon1, lat2, lon2):
    r = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = r*c*1000

    return dist


# Custom adam optimizer
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=10000,
    decay_rate=0.9,
    staircase = False)
adam = optimizers.Adam(learning_rate=lr_schedule)

# evaluate loaded model on test data
loaded_model.compile(loss='mse',
                     optimizer=adam,
                     metrics=[rmse])

# Predict Outputs
prediction = loaded_model.predict(x_test)

print("RMSE (radian per 10 secs):",np.mean(rmse(prediction,y_test)))
print("R2 (radian per 10 secs):",r2_score(y_test[:,0], prediction[:,0]))
error = prediction-y_test

# Save the list as a pickle file
with open('prediction_60ep_2024-04-08-14-01.pkl', 'wb') as f:
    pickle.dump(prediction, f)
    
import matplotlib.pyplot as plt

plt.hist(y_test[:,0], bins = 100, label = "testing_data_0",alpha=0.5)
plt.hist(error[:,0], bins = 100, label = "error",alpha=0.5)
plt.legend() 
plt.grid()
plt.show()

plt.scatter(y_test[:,0], prediction[:,0])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()