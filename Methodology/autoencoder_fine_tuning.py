"""
Fine-tuning, using the spectra observed with CARMENES, of the pre-trained autoencoders
    - phf_good_norm: numpy array containing the normalised synthetic spectra
    - carmf_norm: numpy array containing the normalised CARMENES spectra
    - params_final: list with a dictionary for each of the best combination of hyperparameters found in the grid search
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

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

x_train_carm, x_test_carm = train_test_split(carmf_norm, test_size=0.2, random_state=42)

path = 'path_to_acs'

# Fine-tuning  of the autoencoders
for f in np.sort(os.listdir(path)):
    
    if f.startswith('ac_'):

        num = int(f.split('.')[0].split('_')[1])
        
        lr_model = params_final[num]['lr']
        neurons_in_model = params_final[num]['neurons_in']
        reg_par_model = params_final[num]['reg_par']

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
        
        custom_objects = {"contractive_loss": contractive_loss}
        with keras.utils.custom_object_scope(custom_objects):
            ac = load_model(path + f)
            
        enc = Model(ac.input, ac.layers[6].output)
        
        predicted_ph = ac.predict(phf_good_norm,verbose=0)
        predicted_carm = ac.predict(carmf_norm,verbose=0)
        
        mse_ph = mean_squared_error(predicted_ph,phf_good_norm)
        mse_carm = mean_squared_error(predicted_carm,carmf_norm)
        
        ac.layers[0].trainable = False
        ac.layers[1].trainable = False
        ac.layers[2].trainable = False
        ac.layers[3].trainable = False
        ac.layers[4].trainable = False

        ac.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_model), loss=contractive_loss)
        
        ac.fit(x_train_carm, x_train_carm,
                    epochs=60,
                    batch_size=128,
                    verbose=0,
                    shuffle=True)

        predicted_ph_dtl = ac.predict(phf_good_norm,verbose=0)
        predicted_carm_dtl = ac.predict(carmf_norm,verbose=0)
        predicted_carm_dtl_test = ac.predict(x_test_carm,verbose=0)

        mse_ph_dtl = mean_squared_error(predicted_ph_dtl,phf_good_norm)
        mse_carm_dtl = mean_squared_error(predicted_carm_dtl,carmf_norm)
        mse_carm_dtl_test = mean_squared_error(predicted_carm_dtl_test,x_test_carm)

        f = f.replace('.h5','')
        
        with open('ac_mse_carmdtl.txt','a') as file_mse:
            file_mse.write(f'Autoencoder {f} mse PH: {mse_ph}, mse CARM: {mse_carm}, mse PHDTL: {mse_ph_dtl}, mse CARMDTL: {mse_carm_dtl}, mse CARMDTL TEST: {mse_carm_dtl_test} \n')
            
        encoded_carm = enc.predict(carmf_norm,verbose=0)        
        np.save(f'encoded_carm_{f}.npy', encoded_carm)
        ac.save(f'dtl_{f}.h5')
            
        print(f'AC: {f} finished')
