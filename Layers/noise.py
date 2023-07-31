import tensorflow as tf
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

class noiseLayer(tf.keras.layers.Layer):

    def __init__(self, mean, std, input_shape, indices=[[2, 3], [3, 4]]):
        super(noiseLayer, self).__init__()
        self.mean = mean
        self.std  = std
        shape = [indices[0][1]-indices[0][0], indices[1][1]-indices[1][0]]
        mask = tf.ones(shape) 
        paddings = tf.constant([[indices[0][0], input_shape[0]-indices[0][1]], 
                                [indices[1][0], input_shape[1]-indices[1][1]]])
        self.mask = tf.pad(mask, paddings=paddings)
        self.in_shape = input_shape


    def call(self, input, training=True):

        mean = self.mean
        std  = self.std
        if training == True:
            return input + (tf.random.normal(self.in_shape, 
                                             mean=mean,
                                             stddev=std,
                                             seed=1) * self.mask)
        else:
            return input