import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def MLPRegressor(n_in):
    # Define the MLP model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[n_in]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model


def quantile_loss(q, y, pred):
    err = y - pred
    return tf.keras.backend.mean(tf.keras.backend.maximum(q*err, (q-1)*err), axis=-1)



# mlp quantileregression
def MLPQuantile(n_in, quantiles):

    inputs = keras.Input(shape=n_in)
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)

    outputs = []
    for quantile in quantiles:
        outputs.append(layers.Dense(1)(x))

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=[lambda y, pred: quantile_loss(q, y, pred) for q in quantiles], optimizer="adam")

    return model

