import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LayerNormalization
from tensorflow.keras import Model, datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import copy


class VFLPassiveModel(Model):
    def __init__(self,emb_dim=32):
        super(VFLPassiveModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, name="dense1", activation='relu')
        self.d2 = Dense(emb_dim, name="dense2", activation='relu')
        

    def call(self, x):
        x = self.flatten(x)
        x= self.d1(x)
        x= self.d2(x)
        return x 


class VFLPassiveModelMNIST(Model):
    def __init__(self,emb_dim=32):
        super(VFLPassiveModelMNIST, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, name="dense1", activation='relu')
        self.d2 = Dense(64, name="dense2", activation='relu')
        self.d3 = Dense(emb_dim, name="dense3", activation='relu')

    def call(self, x):
        x = self.flatten(x)
        x= self.d1(x)
        x= self.d2(x)
        x= self.d3(x)
        return x
    

class VFLPassiveModelCIFAR(Model):
    def __init__(self,emb_dim=32):
        super(VFLPassiveModelCIFAR, self).__init__()
        self.d0 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.d1 = layers.MaxPooling2D((2, 2))
        self.d2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.d3 = layers.MaxPooling2D((2,2))
        self.d4 = layers.Conv2D(64, (3, 3), activation='relu')
        self.d5 = layers.Flatten()
        self.d6= layers.Dense(emb_dim, activation='relu')

       

    def call(self, x):       
        x=self.d0(x)
        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.d4(x)
        x=self.d5(x)
        x=self.d6(x)
        return  x