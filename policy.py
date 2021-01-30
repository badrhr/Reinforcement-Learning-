# -*- coding: utf-8 -*-
"""
You can use a simple DNN
"""

 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop

class Policy:
    
    def agent_policy(self,input_shape, action_space):
        X_input = Input(input_shape)
        X = X_input
    
        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)
    
        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
        
        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    
        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    
        model.summary()
        return model
