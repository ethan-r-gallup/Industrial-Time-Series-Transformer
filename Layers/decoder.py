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

class TransformerDecoder(Layer):
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
        print(input_shape)
        self.attention1 = TSMultiHeadAttention(num_heads=self.attention_heads, key_dim=self.dim, name='DecoderAttention1')
        self.dropout1   = Dropout(self.dropout_rate)
        self.norm1     = LayerNormalization(name='decoder_AddNorm1')
        
        self.attention2 = TSMultiHeadAttention(num_heads=self.attention_heads, key_dim=self.dim, name='DecoderAttention2')
        self.dropout2   = Dropout(self.dropout_rate)
        self.norm2     = LayerNormalization(name='decoder_AddNorm3')
        
        self.ff         = FeedForward(layers=self.ff_layers, model_dim=input_shape[-1], name='decoder_ff')
        self.dropout3   = Dropout(self.dropout_rate)
        self.norm3     = LayerNormalization(name='decoder_AddNorm3')
        
        # initialize all layer shapes and weights
        self.call(x, x)
        self.layers = [self.attention1, self.dropout1, self.norm1, 
                       self.attention2, self.dropout2, self.norm2,
                       self.ff, self.dropout3, self.norm3]
        
    def call(self, x, encoder_output):
        
        attention1_output = self.dropout1(self.attention1(x, x, x))
        x                 = self.norm1(attention1_output + x)
        
        attention2_output = self.dropout2(self.attention2(x, encoder_output, encoder_output))
        
        x                 = self.norm2(attention2_output + x)
        ff_output         = self.ff(x)
        out               = self.norm3(ff_output + x)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.attention_heads,
            "key_dim": self.dim,
            "ff_dim": self.ff_dim,
            "dropout1_rate": self.dropout1_rate,
            "dropout2_rate": self.dropout2_rate,
        })
        return config
    
    def summary(self):
        table = []
        for layer in self.layers:
            connections = []
            for node in layer._inbound_nodes:
                # if relevant_nodes and node not in relevant_nodes:
                #     # node is not part of the current network
                #     continue
                for (
                    inbound_layer,
                    node_index,
                    tensor_index,
                    _,
                ) in node.iterate_inbound():
                    connections.append(
                        f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                    )
            table.append([layer.name, layer.output_shape, layer.count_params(), connections])
        # table = [[layer.name, layer.output_shape, layer.count_params()] for layer in self.layers]
        return f"\n{self.name}\n\n{tabulate(table, headers=['Layer', 'Output Shape', 'Param #', 'connections'])}"