"""
Grid search for the best combinations of autoencoder hyperparameters
    - phf_good_norm: numpy array containing the normalised synthetic spectra
    - php_good: numpy array containing the synthetic spectra parameters
    - param_grid: list containing a python dictionary for every grid of parameters to be expored
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import time

x_train, x_test, y_train, y_test = train_test_split(phf_good_norm, php_good, test_size=0.3, random_state=42)

def build_autoencoder(neurons_in=None,reg_par=None,lr=None):
    
    """
    Function to build a flexible autoencoder structure
    """
    
    input_l = Input(shape=(3501,),name='input_l')

    encoded_l1 = Dense(int(neurons_in), activation='relu', name='encoded_1')(input_l) 
    encoded_l2 = Dense(int(neurons_in*3/4), activation='relu', name='encoded_2', activity_regularizer=regularizers.l1(reg_par))(encoded_l1)
    encoded_l3 = Dense(int(neurons_in*2/4), activation='relu', name='encoded_3', activity_regularizer=regularizers.l1(reg_par))(encoded_l2)
    encoded_l4 = Dense(int(neurons_in*1/4), activation='relu', name='encoded_4', activity_regularizer=regularizers.l1(reg_par))(encoded_l3)
    encoded_l5 = Dense(int(neurons_in*1/8), activation='relu', name='encoded_5', activity_regularizer=regularizers.l1(reg_par))(encoded_l4)

    bottleneck = Dense(32, activation='relu', name='bottleneck')(encoded_l5)

    decoded_l1 = Dense(int(neurons_in*1/8), activation='relu', name='decoded_1', activity_regularizer=regularizers.l1(reg_par))(bottleneck)
    decoded_l2 = Dense(int(neurons_in*1/4), activation='relu', name='decoded_2', activity_regularizer=regularizers.l1(reg_par))(decoded_l1)
    decoded_l3 = Dense(int(neurons_in*2/4), activation='relu', name='decoded_3', activity_regularizer=regularizers.l1(reg_par))(decoded_l2)
    decoded_l4 = Dense(int(neurons_in*3/4), activation='relu', name='decoded_4', activity_regularizer=regularizers.l1(reg_par))(decoded_l3)
    decoded_l5 = Dense(int(neurons_in), activation='relu', name='decoded_5', activity_regularizer=regularizers.l1(reg_par))(decoded_l4)

    output_l = Dense(3501, activation='sigmoid', name='output_l')(decoded_l5)

    ac = Model(input_l,output_l)
    
    ac.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    return ac

param_grid = [{'neurons_in':np.linspace(2**10,2**11,5),'reg_par':np.linspace(8e-8,1.2e-7,5),'lr':1e-4}]

# Integrates the autoencoder code into a scikit-learn workflow
autoencoder_base = KerasRegressor(build_autoencoder,verbose=0,neurons_in=param_grid[0]['neurons_in'],reg_par=param_grid[0]['reg_par'],lr=param_grid[0]['lr'])

if 'neurons_in' not in autoencoder_base.get_params().keys() or 'reg_par' not in autoencoder_base.get_params().keys() or 'lr' not in autoencoder_base.get_params().keys(): raise ValueError('Incorrectly defined parameters')

# Performs the grid search
grid = GridSearchCV(estimator=autoencoder_base, param_grid=param_grid, cv=4, verbose=1,scoring='neg_mean_squared_error')

start = time.time()

history = grid.fit(x_train,x_train,
                   epochs=15,
                   batch_size=128)

print(f'Autoencoder Grid Search duration: {(time.time() - start)/60} minutes')

# Prints the results
gs_results = history.cv_results_
print('\n\n CV results:')

for mean_score, params in zip(gs_results['mean_test_score'],gs_results['params']):
    print(mean_score,params)
