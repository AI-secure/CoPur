import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LayerNormalization
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import copy

class QuanVFLActiveModelWithOneLayer(Model):
    def __init__(self, emb_dim=16,class_num=5):
        super(QuanVFLActiveModelWithOneLayer, self).__init__()
    
        self.d1 = Dense(emb_dim, name="dense1", activation='relu')
        self.out = Dense(class_num, name="out", activation='softmax')

    def call(self, x):
        x = self.d1(x)
        return self.out(x)


class VFLActiveModelWithOneLayer(Model):
    def __init__(self,emb_dim=16,class_num=5):
        super(VFLActiveModelWithOneLayer, self).__init__()
        self.concatenated = tf.keras.layers.Concatenate()
        self.d1 = Dense(emb_dim, name="dense1", activation='relu')
        self.out = Dense(class_num, name="out", activation='softmax')

    def call(self, x):
        x = self.concatenated(x)
        x = self.d1(x)
        return self.out(x)
