from tensorflow import keras

def deep_and_wide(hidden_dim=64,activation="relu",dropout=0.3,n_hidden_layers=2,
                regularization=0.001,input_size=21,**kwargs):
    inputs = keras.layers.Input(shape=[input_size], name="input_layer")
    x1 = x2 = inputs
    for i in range(n_hidden_layers):
        x1 = keras.layers.Dense(hidden_dim,activation=activation, name = f"Hidden_{i}",
                            kernel_regularizer=keras.regularizers.l2(regularization))(x1)
        x1 = keras.layers.Dropout(dropout)(x1)
    x = keras.layers.concatenate([x1,x2])
    x = keras.layers.Dense(1,activation="sigmoid")(x)

    model = keras.Model(inputs, x, name="minigooglenet")
    return model