import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LayerNormalization
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import copy
def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, 0.8)
    

class AE_NUS(Model):
    def __init__(self,out_dim = 160 , hidden_dim=200 ,  encode_dim= 150):
        super(AE_NUS, self).__init__()
        self.d1 = Dense(hidden_dim, name="dense1", activation=my_leaky_relu)      
        self.d2 = Dense(encode_dim, name="dense2", activation=None)      
        self.d3 = Dense(hidden_dim, name="dense3", activation=my_leaky_relu)
        self.d4 = Dense(out_dim, name="dense4", activation=None)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x2 =x
        x = self.d3(x2)
        x = self.d4(x)
        return x,x2

class AE(Model):
    def __init__(self,out_dim = 160, encode_dim= 100):
        super(AE, self).__init__()
        self.d1 = Dense(encode_dim, name="dense1", activation=None)      
        self.d2 = Dense(out_dim, name="dense2", activation=None)
    
    def call(self, x):
        x = self.d1(x)
        x2 =x
        x = self.d2(x2)
        return x,x2