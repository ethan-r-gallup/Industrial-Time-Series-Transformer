import sys
from IPython.core import ultratb 
sys.excepthook = ultratb.ColorTB()
from tabulate import tabulate

from keras import backend, utils
from keras.layers import Layer, Dropout, LayerNormalization, Dense, Input
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from Layers.attention import TSMultiHeadAttention
from keras.utils import layer_utils
num = 0

class FeedForward(Layer):
    def __init__(self, layers=[], model_dim=None, ff_dim=8, dropout=[0.1], trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        global num
        self.num = num
        num += 1
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.layers = layers
        self.ff_dim = ff_dim
        self.model_dim = model_dim
        self.dropout1_rate = dropout[0]
        self.dropout2_rate = dropout[1] if (len(dropout) > 1) else self.dropout1_rate
    
    def build(self, input_shape):
        x = Input(input_shape)
        if len(self.layers) < 1:
            self.layers.append(Dense(units=self.ff_dim, name=f'ff_dense1_{self.num}'))
            self.layers.append(Dropout(self.dropout1_rate, name=f'ff_dropout1_{self.num}'))
            self.layers.append(Dense(units=self.model_dim, name=f'ff_dense1_{self.num}'))
            self.layers.append(Dropout(self.dropout2_rate, name=f'ff_dropout2_{self.num}'))
        
        self.call(x)
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    