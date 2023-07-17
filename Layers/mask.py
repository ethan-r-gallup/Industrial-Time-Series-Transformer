import sys
from IPython.core import ultratb 
sys.excepthook = ultratb.ColorTB()
import numpy as np
from tensorflow.math import equal
from einops import rearrange, repeat

from keras import backend
from keras.layers import Layer, Input
import tensorflow as tf

from numpy.random import randint

class Mask(Layer):
    def __init__(self, input_shape, **kwargs):
        
        
    def build(self, input_shape):
        self.shape = input_shape
        self.arr = tf.ones([input_shape[0], input_shape[-2], input_shape[-2]])
    
    def call(self, x, training=None):
        if training is None:
            training = backend.learning_phase()
        if training:
            nums = randint(0, self.shape[-2]-1, self.shape[0])
            mask = tf.sequence_mask(nums, self.shape[-2], dtype=tf.float32)
            x = tf.einsum('btf,bt->btf', x, mask)
        else:
            mask = tf.cast(equal(x[:, :, 0][0], 0), tf.float32)
        
        mask = tf.math.minimum(tf.einsum('bij,bj->bij', self.arr, mask), tf.einsum('bij,bj->bji', self.arr, mask))
        return x, mask
    
    def compute_output_shape(self, input_shape):
        return input_shape, input_shape
        

if __name__ == '__main__':
    mask = Mask()
    data = tf.ones([12, 9, 5])
    print(data.shape)
    inpt = Input([12, 75])
    a, b = mask(data, training=1)
    print(a[0])
    print(b[0])
    
    