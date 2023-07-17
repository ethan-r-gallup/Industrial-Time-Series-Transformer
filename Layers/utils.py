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

def summary(obj):
    table = []
    for layer in obj.layers:
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
    return f"\n{obj.name}\n\n{tabulate(table, headers=['Layer', 'Output Shape', 'Param #', 'connections'])}"