import tensorflow as tf
from tensorflow.keras import layers
from config import Config
from tensorflow.keras import regularizers

def build_model(input_shape, hidden_units, kernel_regularizer, learning_rate, loss_function):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
    ])

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu",
                  kernel_regularizer=regularizers.l2(kernel_regularizer)))

    model.add(layers.Dense(1))

    model.compile(loss=loss_function,
                  optimizer=tf.keras.optimizers.Adam(learning_rate))
    return model
