"""
Training of the CNN regression models. This file shows the training for the effective temperature. 
The process for the other stellar parmeters is the same, including the corresponding hyperparameters for each parameter in the build_cnn() function (Table 1 in the paper)
    - phf_good_norm: numpy array containing the normalised synthetic spectra
    - teff_ph: numpy array containing the effective temperatures for the synthetic spectra
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model


def build_cnn(nf1=None,nf2=None,nu=None):
    
    "
    Builds a flexible 1D CNN structure
    "
    
    model = Sequential()

    model.add(keras.layers.Conv1D(filters=nf1,kernel_size=2, activation='relu', input_shape = (32,1),padding='same'))
    model.add(keras.layers.Conv1D(filters=nf2,kernel_size=2, activation='relu',padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=256,activation='relu'))
    model.add(keras.layers.Dense(units=128,activation='relu'))
    model.add(keras.layers.Dense(units=nu,activation='relu'))

    model.add(keras.layers.Dense(1, activation='linear'))
    
    return model


split = StratifiedShuffleSplit(n_splits=1, test_size=0.3,random_state=42)

path_acs = 'path_to_acs'
path_encodings = 'path_to_latent_representations'

#Training of the CNN models
for f in np.sort(os.listdir(path_acs)):
    
    if f.startswith('ac_'):
        
        num = int(f.split('.')[0].split('_')[1])
        
        ac = f.replace('.h5','')
        
        encoded_ph = np.load(f"{path_encodings}encoded_ph_{ac.replace('_','')}.npy") 
        encoded_carm = np.load(f'{path_encodings}CARMENES/encoded_carm_{ac}.npy') 

        for tr_index, test_index in split.split(encoded_ph,teff_ph):

            x_train = encoded_ph[tr_index]
            y_train = teff_ph[tr_index]

            x_test = encoded_ph[test_index]
            y_test = teff_ph[test_index]
            
        x_train = x_train.reshape(len(x_train),32,1)
        x_test = x_test.reshape(len(x_test),32,1)
        encoded_ph = encoded_ph.reshape(len(encoded_ph),32,1)
        encoded_carm = encoded_carm.reshape(len(encoded_carm),32,1)
        
        #For each autoencoder, trains five different CNN models        
        for i in range(5):
        
            model_cnn = build_cnn(nf1=64,nf2=32,nu=64)            
            model_cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), loss='mean_squared_error')

            model_cnn.fit(x_train,y_train,epochs=200,verbose=0)

            teff_pred = model_cnn.predict(encoded_ph,verbose=0)
            mse_ph = mean_squared_error(teff_pred,teff_ph)

            teff_pred_test = model_cnn.predict(x_test,verbose=0)
            mse_ph_test = mean_squared_error(teff_pred_test,y_test)
        
            with open('cnn_mse_ph_teff.txt','a') as file_mse:
                file_mse.write(f'Autoencoder {ac}_{i}: mse PH: {mse_ph}. mse PH test set: {mse_ph_test} \n')
                
            teff_carm_pred = model_cnn.predict(encoded_carm, verbose=0)
            
            np.save(f'teff_estimation_{i}_{ac}.npy',teff_carm_pred)
        
            model_cnn.save(f'cnn_regressor_{i}_{ac}.h5')
        
        print(f'Autoencoder: {ac} finished')
