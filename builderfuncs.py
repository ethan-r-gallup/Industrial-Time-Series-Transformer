from Layers import TSMultiHeadAttention
from keras.layers import Dense, Dropout, Input, LayerNormalization, Add, GRU, LSTM, Bidirectional
import tensorflow as tf
from tensorflow.math import logical_not, equal
import keras.backend as k
from parameterdicts import TransformerParameters, GRUParameters
from sklearn.metrics import explained_variance_score


import pickle
import json
from numpy import Inf
import numpy as np
from keras.callbacks import Callback
from tensorflow.keras.activations import linear, relu, elu, selu, gelu, sigmoid, tanh

acts = {'linear': linear, 'relu': relu, 'elu': elu, 'selu': selu, 'gelu': gelu, 'sigmoid': sigmoid, 'tanh': tanh}


class EarlyStopAndSave(Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
      filepath: Folder in which to save all the model defining files
  """

    def __init__(self, filepath: str, patience: int = 0, lim: float = 0.5, minormax: str = 'min', quickstop: str = "val_loss"):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.filepath = filepath
        self.lim = lim
        if minormax == 'min':
            self.evalfunc = np.less
        else:
            self.evalfunc = np.greater
        self.minormax = minormax
        self.quickstop = quickstop

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = Inf

        # print(dir(self.model))

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        check_r2 = logs.get(self.quickstop)

        if self.evalfunc(check_r2, self.lim):
            print("stopping", check_r2)
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            print('wait:', self.wait)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        save_whole_model(self.model, self.filepath)
        # if self.stopped_epoch > 0:
            # print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
    

def loss_func(target, prediction):
    """mean square error loss that can take masks in order to work with 
    single-step prediction models

    Args:
        target : true target values
        prediction: predicted target values

    Returns:
        MSE loss
    """
    # Create mask that ignores loss from -1e+9
    mask = logical_not(equal(prediction, -1e+9))

    # calculate square errors
    loss = tf.subtract(prediction, target, name='loss_subtract')
    loss = tf.square(loss, name='loss_square')

    # apply mask and calcualte mean
    loss = tf.ragged.boolean_mask(loss, mask, name='loss_boolean_filter')
    loss = k.mean(loss)
    
    return loss


def explained_variance(target, prediction):
    mask = logical_not(equal(prediction, -1e+9))
    target = tf.ragged.boolean_mask(target, mask)
    prediction = tf.ragged.boolean_mask(prediction, mask)
    ss_res = k.sum(k.square((target-prediction)-k.mean(target-prediction)))
    ss_tot = k.sum(k.square(target - k.mean(target)))
    return 1 - ss_res/(ss_tot)

def r2(target, prediction):
    mask = logical_not(equal(prediction, -1e+9))
    target = tf.ragged.boolean_mask(target, mask)
    prediction = tf.ragged.boolean_mask(prediction, mask)
    ss_res = k.sum(k.square(target-prediction))
    ss_tot = k.sum(k.square(target - k.mean(target)))
    return 1 - ss_res/(ss_tot)


def build_encoder(inputs, idx, name, ff_act, num_heads, key_dim, dropout, ff_dim, dynamic=False):

    # lookback variable self attention
    attention_input_shapes = [inputs.shape]*3
    x = TSMultiHeadAttention(input_shapes=attention_input_shapes,
                             num_heads=num_heads, 
                             key_dim=key_dim,
                             positional_encoding=True,
                             dynamic=dynamic,  
                             name=f"{name}encoder_{idx}._attention")(inputs, inputs, inputs)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}encoder_{idx}.dropout_0")(x)

    # add normalization layer 0
    norm0_out = LayerNormalization(dynamic=dynamic, name=f"{name}encoder_{idx}.norm_0")(Add(dynamic=dynamic)([inputs, x]))
    # Feed-Forward Layers
    x = Dense(units=ff_dim, dynamic=dynamic, name=f"{name}encoder_{idx}.dense_0")(norm0_out)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}encoder_{idx}.dropout_1")(x)
    x = Dense(units=inputs.shape[-1], activation=acts[ff_act], dynamic=dynamic, name=f"{name}encoder_{idx}.dense_1")(norm0_out)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}encoder_{idx}.dropout_2")(x)
    
    # add normalization layer 1
    out = LayerNormalization(dynamic=dynamic, name=f"{name}encoder_{idx}.norm_1")(Add(dynamic=dynamic)([norm0_out, x]))
    return out
    

def build_decoder(inputs, labels, enc_out, mask, idx, name,
                  ff_act, num_heads, key_dim, dropout, ff_dim, dynamic=False):
    
    attention1_input_shapes = [inputs.shape]*3
    attention2_input_shapes = [inputs.shape, enc_out.shape, enc_out.shape]

    """ #< Only uncomment this if you want to do single-step prediction instead of multi-step >#
    attention0_input_shapes = [labels.shape]*3
    attention1_input_shapes = [labels.shape, inputs.shape, inputs.shape]
    attention2_input_shapes = [labels.shape, inputs.shape, inputs.shape]

    x = TSMultiHeadAttention(input_shapes=attention0_input_shapes,
                             num_heads=num_heads, 
                             key_dim=key_dim,
                             positional_encoding=True,
                             dynamic=dynamic, 
                             name=f"{name}decoder_{idx}.attention_0")(labels, labels, labels, attention_mask=mask)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_0")(x)
    norm0_out = LayerNormalization(dynamic=dynamic, name=f"{name}decoder_{idx}.norm_0")(Add(dynamic=dynamic)([labels, x]))

    x = TSMultiHeadAttention(input_shapes=attention1_input_shapes,
                             num_heads=num_heads, 
                             key_dim=key_dim,
                             positional_encoding=True,
                             dynamic=dynamic, 
                             name=f"{name}decoder_{idx}.attention_1")(norm0_out, inputs, inputs, attention_mask=mask)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_1")(x)
    norm1_out = LayerNormalization(dynamic=dynamic, name=f"{name}decoder_{idx}.norm_1")(Add(dynamic=dynamic)([inputs, norm0_out, x]))
    """
    # manipulated variable self attention
    x = TSMultiHeadAttention(input_shapes=attention1_input_shapes,
                             num_heads=num_heads, 
                             key_dim=key_dim,
                             positional_encoding=True,
                             dynamic=dynamic, 
                             name=f"{name}decoder_{idx}.attention_1")(inputs, inputs, inputs, attention_mask=mask)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_1")(x)

    # add normalization layer 1
    norm1_out = LayerNormalization(dynamic=dynamic, name=f"{name}decoder_{idx}.norm_1")(Add(dynamic=dynamic)([inputs, x]))
    
    # cross attention with encoder outputs
    x = TSMultiHeadAttention(input_shapes=attention2_input_shapes,
                             num_heads=num_heads, 
                             key_dim=key_dim,
                             positional_encoding=True,
                             dynamic=dynamic, 
                             name=f"{name}decoder_{idx}.attention_2")(norm1_out, enc_out, enc_out)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_2")(x)
    
    # add normalization layer 2
    norm2_out = LayerNormalization(dynamic=dynamic, name=f"{name}decoder_{idx}.norm_2")(Add(dynamic=dynamic)([norm1_out, x]))
    
    # Feed-Forward Layers
    x = Dense(units=ff_dim, dynamic=dynamic, name=f"{name}decoder_{idx}.dense_0")(norm2_out)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_3")(x)
    x = Dense(units=inputs.shape[-1], activation=acts[ff_act], dynamic=dynamic, name=f"{name}decoder_{idx}.dense_1")(x)
    x = Dropout(dropout, dynamic=dynamic, name=f"{name}decoder_{idx}.dropout_4")(x)
    
    # add normalization layer 3
    out = LayerNormalization(dynamic=dynamic, name=f"{name}decoder_{idx}.norm_3")(Add(dynamic=dynamic)([norm2_out, x]))
    return out


def build_transformer(parameters: TransformerParameters, name: str = 'Transformer', dynamic:bool = False) -> tf.keras.Model:

    # initialize the inputs
    encoder_input = Input(shape=(parameters["look_back"], parameters["n_features"]), 
                          name=f"{name}encoder_input")
    decoder_input = Input(shape=(parameters["horizon"], parameters["n_manips"]),
                          name=f"{name}decoder_input")
    decoder_label = Input(shape=(parameters["horizon"], parameters["n_targs"]),
                          name=f"{name}decoder_label")
    attention_mask = Input(shape=(1, parameters["horizon"], parameters["horizon"]),
                          name=f"{name}attention_mask")
    x_enc, x_dec, y_dec = encoder_input, decoder_input, decoder_label
        
    # Set up the encoder
    for i in range(parameters["num_encoders"]):
        x_enc = build_encoder(x_enc, i, name,
                              parameters["ff_activ"],
                              parameters["num_heads"],
                              parameters["key_dim"],
                              parameters["dropout"],
                              parameters["ff_dim"],
                              dynamic=dynamic)

    mask = tf.cast(attention_mask, tf.bool, name=f"{name}mask_cast1")

    # Set up the decoder
    for i in range(parameters["num_decoders"]):
        x_dec = build_decoder(x_dec, y_dec, x_enc, mask, i, name,
                              parameters["ff_activ"],
                              parameters["num_heads"],
                              parameters["key_dim"],
                              parameters["dropout"],
                              parameters["ff_dim"],
                              dynamic=dynamic)
    
    # Set up the "multi-layer perceptron"
    for i in range(parameters["mlp_layers"]):
        x_dec = Dense(parameters["mlp_units"], activation=acts[parameters["mlp_activ"]], dynamic=dynamic, name=f"{name}mlp_dense{i}")(x_dec)
        x_dec = Dropout(parameters["mlp_dropout"], dynamic=dynamic, name=f"{name}mlp_dropout{i}")(x_dec)

    # Define the final dense layer
    outputs = Dense(parameters["n_targs"], activation=acts[parameters["out_activ"]], dynamic=dynamic, name=f"{name}output_layer")(x_dec)
    outputs = tf.where(mask[:, 0, :, 0:parameters["n_targs"]], -1e+9, outputs)

    # build the model
    model = tf.keras.Model(inputs={'encoder_inputs':encoder_input, 'decoder_inputs':decoder_input, 'decoder_labels': decoder_label, 'attention_mask': attention_mask}, outputs=outputs, name=name)

    # set up the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"],
                                         beta_1=parameters["beta_1"],
                                         beta_2=parameters["beta_2"],
                                         epsilon=parameters["epsilon"])

    # compile the model
    model.compile(optimizer=optimizer, loss=[loss_func], metrics=['mse', 'mae', r2, explained_variance])
    model.build_dict = parameters
    return model


def build_gru(parameters: GRUParameters, name: str = 'gru_model', dynamic:bool = False) -> tf.keras.Model:
    model_inputs = Input(shape=(parameters["look_back"], parameters["n_features"]),
                         name='gru_input')
    x = GRU(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=True,
            dynamic=dynamic,
            name='gru_0')(model_inputs)
    x = GRU(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=False,
            dynamic=dynamic,
            name='gru_1')(x)
    model_outputs = Dense(units=parameters["n_targs"], dynamic=dynamic, name="output")(x)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"],
                                         beta_1=parameters["beta_1"],
                                         beta_2=parameters["beta_2"],
                                         epsilon=parameters["epsilon"])

    # compile the model
    model.compile(optimizer=optimizer, loss=[loss_func])
    model.build_dict = parameters
    return model


def build_bigru(parameters: GRUParameters, name: str = 'bigru_model', dynamic:bool = False) -> tf.keras.Model:
    model_inputs = Input(shape=(parameters["look_back"], parameters["n_features"]),
                         name='bigru_input')
    x = Bidirectional(GRU(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=True,
            dynamic=dynamic,
            name='bigru_0'))(model_inputs)
    x = Bidirectional(GRU(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=False,
            dynamic=dynamic,
            name='bigru_1'))(x)
    model_outputs = Dense(units=parameters["n_targs"], dynamic=dynamic, name="output")(x)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"],
                                         beta_1=parameters["beta_1"],
                                         beta_2=parameters["beta_2"],
                                         epsilon=parameters["epsilon"])

    # compile the model
    model.compile(optimizer=optimizer, loss=[loss_func])
    model.build_dict = parameters
    return model


def build_lstm(parameters: GRUParameters, name: str = 'lstm_model', dynamic:bool = False) -> tf.keras.Model:
    model_inputs = Input(shape=(parameters["look_back"], parameters["n_features"]),
                         name='lstm_input')
    x = LSTM(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=True,
            dynamic=dynamic,
            name='lstm_0')(model_inputs)
    x = LSTM(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=False,
            dynamic=dynamic,
            name='lstm_1')(x)
    model_outputs = Dense(units=parameters["n_targs"], dynamic=dynamic, name="output")(x)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"],
                                         beta_1=parameters["beta_1"],
                                         beta_2=parameters["beta_2"],
                                         epsilon=parameters["epsilon"])

    # compile the model
    model.compile(optimizer=optimizer, loss=[loss_func])
    model.build_dict = parameters
    return model


def build_bilstm(parameters: GRUParameters, name: str = 'bilstm_model', dynamic:bool = False) -> tf.keras.Model:
    model_inputs = Input(shape=(parameters["look_back"], parameters["n_features"]),
                         name='bilstm_input')
    x = Bidirectional(LSTM(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=True,
            dynamic=dynamic,
            name='bilstm_0'))(model_inputs)
    x = Bidirectional(LSTM(units=parameters["units"],
            activation=parameters["activation"],
            recurrent_activation=parameters["recurrent_activation"],
            recurrent_dropout=parameters["recurrent_dropout"],
            dropout=parameters["dropout"],
            return_sequences=False,
            dynamic=dynamic,
            name='bilstm_1'))(x)
    model_outputs = Dense(units=parameters["n_targs"], dynamic=dynamic, name="output")(x)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"],
                                         beta_1=parameters["beta_1"],
                                         beta_2=parameters["beta_2"],
                                         epsilon=parameters["epsilon"])

    # compile the model
    model.compile(optimizer=optimizer, loss=[loss_func])
    model.build_dict = parameters
    return model



def save_whole_model(model, filepath):
        model.save_weights(f"{filepath}/weights.h5")
        with open(f"{filepath}/parameters.json", "w") as f:
            json.dump(model.build_dict, f)
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = k.batch_get_value(symbolic_weights)
        with open(f"{filepath}/optimizer.pkl", "wb") as f:
            pickle.dump(weight_values, f)


def restore_model(filepath, modeltype):
    with open(f"{filepath}/parameters.json") as f:
        parameters = json.load(f)
    if modeltype == "transformer":
        model = build_transformer(parameters)
    elif modeltype == "gru":
        model = build_gru(parameters)
    elif modeltype == "bigru":
        model = build_bigru(parameters)
    elif modeltype == "lstm":
        model = build_lstm(parameters)
    elif modeltype == "bilstm":
        model = build_bilstm(parameters)
    model.load_weights(f"{filepath}/weights.h5")
    with open(f"{filepath}/optimizer.pkl", "rb") as f:
        weight_values = pickle.load(f)

    # use fake gradient to let optimizer init the wieghts.
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))

    # This will work because the new optimizer is initialized.
    model.optimizer.set_weights(weight_values)
    return model




if __name__ == '__main__':
    import json
    with open(f"GRUparams.json") as f:
        parameters = json.load(f)
    model = build_gru(parameters)
    print(model.summary())
    