#!/usr/bin/env python

"""
Title: danish_model.py
Objective: Create a Keras model and Train with AIS Data
Creator: Stig Terrebonne
Date: July 27th, 2018
"""

from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras import optimizers
from keras import backend
import numpy as np
import matplotlib.pyplot as plt

# Create date string to use in the naming of model files
datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')


# Function Definitions
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# Constants
INCLUDE_DROPOUT = False         # Include dropout layer?
DROPOUT_VALUE = 0.2             # Ratio of neurons to dropout
SAVEFIGS = True                 # Save Training Figures?
ACTIVATION_LAYER = 'linear'     # Activation function on Dense Layer (if 'none'), no activation
LEAKY = True                    # Include a Leaky RelU Activation Layer?
NUM_EPOCHS = 60                 # Number of Epochs to Train Over
BATCH_SIZE = 2                  # Number of batches

num_features = 6       # The data we are submitting per time step (lat, long, speed, time, course)
num_timesteps = 18       # The number of time steps per sequence (track)

# Pull Data From Numpy File
import pickle

# Open the pickle file in binary read mode ('rb')
with open('dataset/filtered_samso_lstm_18_ts_one_out.pkl', 'rb') as f:
    # Load data from the pickle file
    loaded_data = pickle.load(f)
x_train = loaded_data['X_train']
y_train = loaded_data['Y_train']
# Defining Layers
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(num_timesteps, num_features)))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(8))
if LEAKY:
    model.add(LeakyReLU(alpha=0.01))
if INCLUDE_DROPOUT:
    model.add(Dropout(DROPOUT_VALUE))
if ACTIVATION_LAYER == 'none':
    model.add(Dense(2))
else:
    model.add(Dense(2, activation=ACTIVATION_LAYER))

# Custom adam optimizer
#adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=10000,
    decay_rate=0.9,
    staircase = False)
adam = optimizers.Adam(learning_rate=lr_schedule)
# Compile Model, with custom loss and optimizer
model.compile(loss='mse',
              optimizer=adam,
              metrics=[rmse])

# Plot the Model to Image File
# plot_model(model, to_file='..\..\models\danish_model_' + ACTIVATION_LAYER + '_' + str(NUM_EPOCHS) + 'ep_' + datestring + '.png', show_shapes=True)
print(model.summary())

# Fit the Model
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_split=0.20)

# Evaluate the Model
scores = model.evaluate(x_train, y_train)
print('\n%s: %f\n' % (model.metrics_names[1], scores[1]))

# Serialize Model to JSON
model_json = model.to_json()
with open('dataset\models\samso_model_' + ACTIVATION_LAYER + '_' + str(NUM_EPOCHS) + 'ep_' + datestring + '.json', 'w') as json_file:
    json_file.write(model_json)

# Serialize Weights to HDF5
model.save_weights('dataset\models\samso_model_' + ACTIVATION_LAYER + '_' + str(NUM_EPOCHS) + 'ep_' + datestring + '.h5')
print('Saved model to file location: dataset/models')

plt.figure(0)
plt.plot(history.history['rmse'])
plt.title('Danish: Regression Analytics')
plt.xlabel('Epoch')
plt.legend(['RMSE'
            # 'MSE',
            # 'MAE',
            # 'cos Prox'
            ], loc='upper right')
if SAVEFIGS:
    plt.savefig('result/figures/samso_rmse_' + ACTIVATION_LAYER + '_' + str(NUM_EPOCHS) + 'ep_' + datestring + '.png', bbox_inches='tight')
plt.show(block=False)
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Samso: Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
if SAVEFIGS:
    plt.savefig('result/figures/samso_loss_' + ACTIVATION_LAYER + '_' + str(NUM_EPOCHS) + 'ep_' + datestring + '.png', bbox_inches='tight')
plt.show()
