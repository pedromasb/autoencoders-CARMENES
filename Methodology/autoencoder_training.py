"""
Training of the autoencoders for the best hyperparameters combinations
    - phf_good_norm: numpy array containing the normalised synthetic spectra
    - php_good: numpy array containing the synthetic spectra parameters
    - params_final: list with a dictionary for each of the best combination of hyperparameters found in the grid search.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers

def contractive_loss(y_pred, y_true):
    
    """
    Implements the contractive loss term
    """

    lm = 1e-4

    MSE = K.mean(K.square(y_true - y_pred), axis=1)

    weights = K.variable(value=ac.get_layer('bottleneck').get_weights()[0]) 
    weights = K.transpose(weights) 

    h = ac.get_layer('bottleneck').output

    penalty_term =  K.sum(((h * (1 - h))**2) * K.sum(weights**2, axis=1), axis=1)

    Loss = MSE + (lm * penalty_term)

    return Loss

x_train, x_test, y_train, y_test = train_test_split(phf_good_norm, php_good, test_size=0.3, random_state=42)

# Training of the autoencoders
for i in range(len(params_final)):
    
    lr_model = params_final[i]['lr']
    neurons_in_model = params_final[i]['neurons_in']
    reg_par_model = params_final[i]['reg_par']

    input_l = Input(shape=(3501,),name='input_l')

    encoded_l1 = Dense(int(neurons_in_model), activation='relu', name='encoded_1')(input_l) 
    encoded_l2 = Dense(int(neurons_in_model*3/4), activation='relu', name='encoded_2',activity_regularizer=regularizers.l1(reg_par_model))(encoded_l1)
    encoded_l3 = Dense(int(neurons_in_model*2/4), activation='relu', name='encoded_3',activity_regularizer=regularizers.l1(reg_par_model))(encoded_l2)
    encoded_l4 = Dense(int(neurons_in_model*1/4), activation='relu', name='encoded_4',activity_regularizer=regularizers.l1(reg_par_model))(encoded_l3)
    encoded_l5 = Dense(int(neurons_in_model*1/8), activation='relu', name='encoded_5',activity_regularizer=regularizers.l1(reg_par_model))(encoded_l4)

    bottleneck = Dense(32, activation='relu', name='bottleneck')(encoded_l5)

    decoded_l1 = Dense(int(neurons_in_model*1/8), activation='relu', name='decoded_1',activity_regularizer=regularizers.l1(reg_par_model))(bottleneck)
    decoded_l2 = Dense(int(neurons_in_model*1/4), activation='relu', name='decoded_2',activity_regularizer=regularizers.l1(reg_par_model))(decoded_l1)
    decoded_l3 = Dense(int(neurons_in_model*2/4), activation='relu', name='decoded_3',activity_regularizer=regularizers.l1(reg_par_model))(decoded_l2)
    decoded_l4 = Dense(int(neurons_in_model*3/4), activation='relu', name='decoded_4',activity_regularizer=regularizers.l1(reg_par_model))(decoded_l3)
    decoded_l5 = Dense(int(neurons_in_model), activation='relu', name='decoded_5',activity_regularizer=regularizers.l1(reg_par_model))(decoded_l4)

    output_l = Dense(3501, activation='sigmoid', name='output_l')(decoded_l5)

    ac = Model(input_l,output_l) 
    encoder = Model(input_l,bottleneck)
    
    ac.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_model), loss=contractive_loss)
    
    ac.fit(x_train,x_train,
                    epochs=120,
                    batch_size=128,
                    verbose=0)
                    
    predicted_ph_test = ac.predict(x_test)
    
    mse = mean_squared_error(predicted_ph_test,x_test)
        
    with open('ac_mse_ph.txt','a') as file_mse:
        file_mse.write(f'Autoencoder ac_{i} mse in TEST SET: {mse} \n')  
    
    encoded_ph = encoder.predict(phf_good_norm)
    np.save(f'encoded_ph_ac{i}.npy', encoded_ph)
    ac.save(f'ac_{i}.h5')
        
    print(f'Autoencoder: {i} finished')
