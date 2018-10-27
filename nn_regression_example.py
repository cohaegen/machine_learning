# Simple regression example of two output variables using a keras neural net

import numpy as np
import tensorflow as tf
import keras
import keras.models
import keras.layers
import keras.regularizers
import matplotlib.pyplot as plt

def create_model():
    '''Return a keras model to be used for regression. It has one input node
    and one output node'''
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, input_dim=1, kernel_initializer='normal', activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(2, kernel_initializer='normal', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    

def create_dataset():
    '''Return a dataset (X, Y) such that Y = 2*X plus noise'''
    X = np.random.random(1000)*100
    Y = np.array([2*X + np.random.random(1000)*2, 3*X + np.random.random(1000)*2]).transpose()
    #Y = Y.reshape(-1,1)
    return (X, Y)
    
def main():
    model = create_model()
    X, Y = create_dataset()
    X_test, y_test = create_dataset()
    model.fit(X, Y, epochs=50)
    plt.scatter(X_test, y_test[:,0],marker='.')
    plt.scatter(X_test, model.predict(X_test)[:,0], marker='.')
    plt.scatter(X_test, y_test[:,1],marker='.')
    plt.scatter(X_test, model.predict(X_test)[:,1], marker='.')
    plt.legend(['Y actual [0]', 'Y predicted [0]', 'Y actual [1]', 'Y predicted [1]'])
    plt.show()