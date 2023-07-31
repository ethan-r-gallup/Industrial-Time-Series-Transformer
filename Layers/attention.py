import sys
from IPython.core import ultratb 
sys.excepthook = ultratb.ColorTB()

from einops import rearrange

from tensorflow.keras.layers import Layer, Input, EinsumDense
import tensorflow as tf

class TSMultiHeadAttention(Layer):
    def __init__(self, input_shapes=None, num_heads=10, key_dim=None, positional_encoding=False, activation=None,
                 trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        # self._dim = (int(input_size / num_heads)) if key_dim is None else key_dim
        self.input_shapes = {"q": input_shapes[0],
                             "k": input_shapes[1],
                             "v": input_shapes[2]} if input_shapes else None
        self.activation = activation
        self._dim = key_dim
        self._num_heads = num_heads
        self.positional_encoding = positional_encoding
        
        
    def build(self, input_shape):
        if not self.input_shapes:
            # if no shape was specified, use inputs to define shape
            self.input_shapes = {"q": input_shape,
                                 "k": input_shape,
                                 "v": input_shape}
        
        # build inputs
        query = Input(shape=self.input_shapes["q"][-2:], name='query')
        key   = Input(shape=self.input_shapes["k"][-2:], name='key')
        value = Input(shape=self.input_shapes["v"][-2:], name='value')
        
        # redefine key_dim just in case the user forgot to
        self._dim = max(int(input_shape[-1] / self._num_heads), 4) if self._dim is None else self._dim
        
        # build Layers
        # q[batches, tokens, features], Weights[features, heads, dim] -> Q[batches, tokens, heads, dim]
        self.q_dense = EinsumDense('btf,fhd->bthd', output_shape=[input_shape[-2], self._num_heads, self._dim], bias_axes='hd', name=f'{self.name}_qdense')
        # k[batches, tokens, features], Weights[features, heads, dim] -> K[batches, tokens, heads, dim]
        self.k_dense = EinsumDense('btf,fhd->bthd', output_shape=[input_shape[-2], self._num_heads, self._dim], bias_axes='hd', name=f'{self.name}_kdense')
        # v[batches, tokens, features], Weights[features, heads, dim] -> V[batches, tokens, heads, dim]
        self.v_dense = EinsumDense('btf,fhd->bthd', output_shape=[input_shape[-2], self._num_heads, self._dim], bias_axes='hd', name=f'{self.name}_vdense')
        if self.positional_encoding:
            # Q[batches, tokens, heads, dim], Weights[features, dim] -> QR[batches, heads, tokens, tokens]
            self.qr_dense = EinsumDense('bihd,jd->bhij', output_shape=[self._num_heads, input_shape[-2], input_shape[-2]], bias_axes='hij', name=f'{self.name}_qrdense')
        # out[batches, tokens, heads, dim], Weights[heads, dim, features] -> output[batches, tokens, features]
        self.Wo = EinsumDense('bthd,hdf->btf', output_shape=[input_shape[-2], input_shape[-1]], bias_axes='f', name=f'{self.name}_Wo')
        
        # initialize all layer shapes and weights
        self.call(query, key, value)
        self.layers = [self.q_dense, self.k_dense, self.v_dense, self.Wo]
    
    def call(self, q, k, v, attention_mask=None, value_mask=None):
        # generate query key and value matrices
        Q = self.q_dense(q)
        K = self.k_dense(k)
        V = self.v_dense(v)
        
        # Q[batches, tokens, heads, dim], K[bathces, tokens, heads, dim] -> QK[batches, heads, tokens, tokens]
        QK = tf.einsum('bihd, bjhd -> bhij', Q, K)

        # apply positional encoding
        if self.positional_encoding:
            QR = self.qr_dense(Q)
            QK += QR
        
        # calculate attention
        QK_d = tf.divide(QK, tf.sqrt(float(self._dim)))
        
        # apply masks
        if attention_mask is not None:
            QK_d = tf.where(attention_mask, QK_d, -1e+9)
        if value_mask is not None:
            V = tf.where(value_mask, V, 0)
            key_mask=rearrange(value_mask, '1 t 1 1 -> 1 1 1 t')
            QK_d = tf.where(key_mask, QK_d, -1e+9)
        
        # get softmax attentions scores
        attention = tf.nn.softmax(QK_d)
        
        # attention[batches, heads, tokens, tokens], V[bathces, tokens, heads, dim] -> out[bathces, tokens, heads, dim]
        out = tf.einsum('bhij, bjhd->bihd', attention, V)
        
        out = self.Wo(out)
        return out   
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._dim,
            "positional_encoding": self.positional_encoding
        })
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape

    


if __name__ == '__main__':
    from mask import Mask
    import numpy as np
    np.set_printoptions(edgeitems=30, linewidth=100000, 
                        formatter=dict(float=lambda x: "%.3g" % x))
    mask = Mask()
    attention = TSMultiHeadAttention(num_heads=3, positional_encoding=False)
    data = tf.ones((1, 11, 12))
    m, vm = mask(data, training=1)
    print(m)
    out = attention(data, data, data, attention_mask=tf.cast(vm, tf.bool))
    # out = attention(data, data, data, attention_mask=m)

    print(out)
    # for var in attention.trainable_weights:
    #     print(var)
