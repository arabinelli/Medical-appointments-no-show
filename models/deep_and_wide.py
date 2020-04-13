import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import Hyperband


class DeepAndWide(keras.Model):

    def __init__(self,hidden_dim=64,activation="relu",dropout=0.3,n_hidden_layers=2,regularization=0.001,**kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(hidden_dim,activation=activation, 
                                        kernel_regularizer=keras.regularizers.l2(regularization),name="Hidden")
        self.output_layer = keras.layers.Dense(1,activation="sigmoid")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.n_hidden_layers = n_hidden_layers
    
    def call(self,inputs,training=False):
        inputs
        for _ in range(self.n_hidden_layers):
            x2 = self.hidden(inputs)
            x2 = self.dropout(x2, training=training)
        x = keras.layers.concatenate([inputs,x2])
        output = self.output_layer(x)
        return output
