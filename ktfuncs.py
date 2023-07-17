import keras_tuner as kt
import tensorflow as tf
from parameterdicts import *
from builderfuncs import build_transformer, build_gru, build_bigru, build_lstm, build_bilstm, acts



def kt_transformer(hp: kt.HyperParameters) ->tf.keras.Model:
    """This function takes inputs from keras tuner, generates the hyperparameter
    dict returns the cooresponding build_transformer result.

    Args:
        hp keras_tuner.HyperParameters: object containing hyperparameters

    Returns:
        tf.keras.Model: built transformer model
    """

    # calculate the internal dimensioality of the attention mechanism
    n_heads = hp.Int("num_heads", min_value=4, max_value=16, step=2) 
    k_dim = 248//n_heads

    # build parameter dictionary
    parameter_space:TransformerParameters = {
        "look_back":     hp.Fixed("lookback", 12),
        "n_features":    hp.Fixed("n_features", 247),
        "n_manips":      hp.Fixed("n_manips", 74),
        "n_targs":       hp.Fixed("n_targs", 1),
        "horizon":       hp.Fixed("horizon", 12),
        
        "num_heads":     n_heads,
        "key_dim":       k_dim,
        "ff_dim":        hp.Int("ff_dim", min_value=8, max_value=16, step=4),
        "ff_activ":      hp.Choice("ff_activ", list(acts.keys())),
        "num_encoders":  hp.Int("num_encoders", min_value=1, max_value=2, step=1),
        "num_decoders":  hp.Int("num_decoders", min_value=1, max_value=6, step=1),
        "mlp_layers":    hp.Int("mlp_layers", min_value=1, max_value=4, step=1),
        "mlp_units":     hp.Int("mlp_units", min_value=64, max_value=256, step=64),
        "mlp_activ":     hp.Choice("mlp_activ", list(acts.keys())),
        "mlp_dropout":   hp.Float("mlp_dropout", min_value=0.1, max_value=.7, step=0.1),
        "dropout":       hp.Float("dropout", min_value=0.1, max_value=.7, step=0.1),
        "out_activ":     hp.Choice("out_activ", list(acts.keys())),
        
        "learning_rate": hp.Fixed('learning_rate', 0.0001),
        "beta_1":        hp.Fixed("beta_1", 0.85),
        "beta_2":        hp.Float("beta_2", min_value=0.99, max_value=0.999, step=0.001),
        "epsilon":       hp.Fixed("epsilon", 1e-7)
    }
    
    return build_transformer(parameter_space)


def kt_gru(hp: kt.HyperParameters) ->tf.keras.Model:
    parameter_space:GRUParameters = {
        "look_back":            hp.Fixed("lookback", 12),
        "n_features":           hp.Fixed("n_features", 247),
        "n_manips":             hp.Fixed("n_manips", 74),
        "n_targs":              hp.Fixed("n_targs", 174),
        "horizon":              hp.Fixed("horizon", 12),
        
        "units":                hp.Int('units', min_value=108, max_value=556, step=64),
        "recurrent_dropout":    hp.Fixed("recurrent_dropout", 0.0),
        "dropout":              hp.Float("dropout", min_value=0.1, max_value=.5, step=0.1),
        "activation":           hp.Fixed("activation", 'tanh'),
        "recurrent_activation": hp.Fixed("recurrent_activation", 'sigmoid'),
        
        "learning_rate":        hp.Fixed('learning_rate', 0.0001),
        "beta_1":               hp.Fixed("beta_1", 0.85),
        "beta_2":               hp.Fixed("beta_2", 0.995),
        "epsilon":              hp.Fixed("epsilon", 1e-7)
    }

    return build_gru(parameter_space)


def kt_bigru(hp: kt.HyperParameters) ->tf.keras.Model:
    parameter_space:GRUParameters = {
        "look_back":            hp.Fixed("lookback", 12),
        "n_features":           hp.Fixed("n_features", 247),
        "n_manips":             hp.Fixed("n_manips", 74),
        "n_targs":              hp.Fixed("n_targs", 174),
        "horizon":              hp.Fixed("horizon", 12),
        
        "units":                hp.Int('units', min_value=108, max_value=556, step=64),
        "recurrent_dropout":    hp.Fixed("recurrent_dropout", 0.0),
        "dropout":              hp.Float("dropout", min_value=0.1, max_value=.5, step=0.1),
        "activation":           hp.Fixed("activation", 'tanh'),
        "recurrent_activation": hp.Fixed("recurrent_activation", 'sigmoid'),
        
        "learning_rate":        hp.Fixed('learning_rate', 0.0001),
        "beta_1":               hp.Fixed("beta_1", 0.85),
        "beta_2":               hp.Fixed("beta_2", 0.995),
        "epsilon":              hp.Fixed("epsilon", 1e-7)
    }

    return build_bigru(parameter_space)

def kt_lstm(hp: kt.HyperParameters) ->tf.keras.Model:
    parameter_space:GRUParameters = {
        "look_back":            hp.Fixed("lookback", 12),
        "n_features":           hp.Fixed("n_features", 247),
        "n_manips":             hp.Fixed("n_manips", 74),
        "n_targs":              hp.Fixed("n_targs", 174),
        "horizon":              hp.Fixed("horizon", 12),
        
        "units":                hp.Int('units', min_value=108, max_value=556, step=64),
        "recurrent_dropout":    hp.Fixed("recurrent_dropout", 0.0),
        "dropout":              hp.Float("dropout", min_value=0.1, max_value=.5, step=0.1),
        "activation":           hp.Fixed("activation", 'tanh'),
        "recurrent_activation": hp.Fixed("recurrent_activation", 'sigmoid'),
        
        "learning_rate":        hp.Fixed('learning_rate', 0.0001),
        "beta_1":               hp.Fixed("beta_1", 0.85),
        "beta_2":               hp.Fixed("beta_2", 0.995),
        "epsilon":              hp.Fixed("epsilon", 1e-7)
    }

    return build_lstm(parameter_space)

def kt_bilstm(hp: kt.HyperParameters) ->tf.keras.Model:
    parameter_space:GRUParameters = {
        "look_back":            hp.Fixed("lookback", 12),
        "n_features":           hp.Fixed("n_features", 247),
        "n_manips":             hp.Fixed("n_manips", 74),
        "n_targs":              hp.Fixed("n_targs", 174),
        "horizon":              hp.Fixed("horizon", 12),
        
        "units":                hp.Int('units', min_value=108, max_value=556, step=64),
        "recurrent_dropout":    hp.Fixed("recurrent_dropout", 0.0),
        "dropout":              hp.Float("dropout", min_value=0.1, max_value=.5, step=0.1),
        "activation":           hp.Fixed("activation", 'tanh'),
        "recurrent_activation": hp.Fixed("recurrent_activation", 'sigmoid'),
        
        "learning_rate":        hp.Fixed('learning_rate', 0.0001),
        "beta_1":               hp.Fixed("beta_1", 0.85),
        "beta_2":               hp.Fixed("beta_2", 0.995),
        "epsilon":              hp.Fixed("epsilon", 1e-7)
    }

    return build_bilstm(parameter_space)