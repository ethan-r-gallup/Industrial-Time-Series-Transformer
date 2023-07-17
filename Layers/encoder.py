import sys
from IPython.core import ultratb 
sys.excepthook = ultratb.ColorTB()
from tabulate import tabulate

from keras import backend, utils
from keras.layers import Layer, Dropout, LayerNormalization, Dense, Input
import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export
from Layers.attention import TSMultiHeadAttention
from Layers.feedforward import FeedForward
from keras.utils import layer_utils


class TransformerEncoder(Layer):
    def __init__(self, ff_layers=[], num_heads=10, key_dim=None, dropout=0.1, 
                 ff_dim=8, trainable=True, name=None, 
                 dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.attention_heads = num_heads
        self.dim = key_dim
        self.ff_dim = ff_dim
        self.ff_layers = ff_layers
        self.dropout_rate = dropout

    def build(self, input_shape):
        x = Input(input_shape[1:])
        
        # Initialize Layers
        self.attention = TSMultiHeadAttention(num_heads=self.attention_heads, key_dim=self.dim, name='Attention')
        self.dropout1  = Dropout(self.dropout_rate)
        self.norm1    = LayerNormalization(name='encoder_norm1')
        
        self.ff       = FeedForward(layers=self.ff_layers, model_dim=input_shape[-1], name='encoder_ff')
        self.norm2    = LayerNormalization(name='encoder_norm2')
        
        # initialize all layer shapes and weights
        self.call(x)
        self.layers = [self.attention, self.dropout1, self.norm1, self.ff, self.norm2]
        
    def call(self, x):
        attention_output = self.dropout1(self.attention(x, x, x))
        x                = self.norm1(attention_output + x)
        ff_output        = self.ff(x)
        out              = self.norm2(ff_output + x)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.attention_heads,
            "key_dim": self.dim,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "dropout1_rate": self.dropout1_rate,
            "dropout2_rate": self.dropout2_rate,
        })
        return config


if __name__ == '__main__':
    import einops
    import numpy as np
    # from encoder import TransformerEncoder
    import tensorflow as tf
    arr = np.arange(0, 24, dtype=float)
    # arr = np.ones(24)
    arr = einops.rearrange(arr, '(b t d) -> b t d', b=2, t=3)
    # print(arr)
    arr = tf.constant(arr)
    encoder = TransformerEncoder(num_heads=2)
    arr_out = encoder(arr)
    print(encoder.summary())
        
    
    
        
        