from tensorflow import keras

class DeepAndWide(keras.Model):
    """
    Defines and initializes the deep and wide model using the Tensorflow 2.0 Keras Subclassing API
    """

    def __init__(self,hidden_dim=64,activation="relu",dropout=0.3,n_hidden_layers=2,
                regularization=0.001,**kwargs):
        """
        Defines the deep and wide network model

        ARGUMENTS:
        hidden_dim (int - defaults to 64): the number of neurons of the hidden layers
        activation (string - defaults to "relu"): the activation of the hidden layers
        dropout (float - defaults to 0.3): the percentage of neurons that will drop out during training
                                           to prevent overfitting
        n_hidden_layers (int - defaults to 2): the number of hidden layers of the model
        regularization (float - defaults to 0.001): the L2 regularization score used to prevent overfitting

        RETURNS:
        Model (tensorflow.keras.Model) to be compiled and fit
        """
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(hidden_dim,activation=activation, 
                                        kernel_regularizer=keras.regularizers.l2(regularization))
        self.output_layer = keras.layers.Dense(1,activation="sigmoid")
        self.dropout = keras.layers.Dropout(dropout)
        self.n_hidden_layers = n_hidden_layers
    
    def call(self,inputs,training=False):
        """
        Defines the model architecture
        """
        x2 = self.hidden(inputs)
        x2 = self.dropout(x2, training=training)
        for _ in range(self.n_hidden_layers - 1):
            x2 = self.hidden(x2)
            x2 = self.dropout(x2, training=training)
        x = keras.layers.concatenate([inputs,x2])
        output = self.output_layer(x)
        return output
