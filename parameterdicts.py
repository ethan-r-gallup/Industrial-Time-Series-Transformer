from typing import TypedDict

class TransformerParameters(TypedDict):
    look_back:    int  # number of timesteps to look back
    n_features:   int  # number of variables in the data
    n_manips:     int  # number of variable to input to the decoder
    n_targs:      int  # number of prediction targets
    horizon:      int  # how many timesteps to look ahead
    
    num_heads:    int  # number of attention heads
    key_dim:      int  # internal dimensionality of the attention mechanism
    ff_dim:       int  # number of nodes in the feedforward layers within the encoders and decoders
    ff_activ:     str  # activation of the feedforward networks within the encoders and decoders
    num_encoders: int  # number of sequential encoders to build
    num_decoders: int  # number of sequential decoders to build
    mlp_layers:   int  # number of layers in the multilayer perceptron that feeds the output
    mlp_units:    int  # number of nodes in each multilayer perceptron layer
    mlp_activ:    str  # activation of the multilayer perceptron network
    mlp_dropout:  float  # dropout rate after each multiperceptron layer
    dropout:      float  # dropout rate after each feedforward layer in the encoders and decoders
    out_activ:    str  # activation of the output layer

    learning_rate: float
    beta_1:        float
    beta_2:        float
    epsilon:       float

class GRUParameters(TypedDict):
    look_back:    int  # number of timesteps to look back
    n_features:   int  # number of variables in the data
    n_manips:     int  # number of variable to input to the decoder
    n_targs:      int  # number of prediction targets
    horizon:      int  # how many timesteps to look ahead
    
    units:                int  # number of units in each cell
    recurrent_dropout:    float  # recurrent dropout applied within GRU cell
    dropout:              float  # dropout rate after each GRU layer
    activation:           str  # activation of the GRU layers
    recurrent_activation: str  # recurrent activation within each GRU cell

    learning_rate: float
    beta_1:        float
    beta_2:        float
    epsilon:       float