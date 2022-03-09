'''
Conv-mixer implementation
from (https://keras.io/examples/vision/convmixer/)
'''

from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def construct_convmixer_model(include_classification=True, nclasses=10, **parameters):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """

    kernel_size = parameters['kernel_size']
    spectrogram_dim = parameters['spectrogram_dim']
    filters = parameters['nfilters']
    top_flatten = parameters['top_flatten']
    dropout = parameters['dropout']
    pooling = parameters['pooling']
    verbose = parameters['verbose']
    patch_size = parameters['patch_size']

    inputs = keras.Input(shape=spectrogram_dim)
    #x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters[0], patch_size)

    # ConvMixer blocks.
    for i, nfilter in enumerate (filters):
        x = conv_mixer_block(x, nfilter, kernel_size)
        x = layers.MaxPool2D(pool_size=pooling[i])(x)
        x = layers.Dropout(rate=dropout[i])(x)

    # Classification block.
    if top_flatten == 'avg':
         x = layers.GlobalAveragePooling2D()(x)
    elif top_flatten == 'max':
         x = layers.GlobalMaxPooling2D()(x)
    #x = layers.GlobalAvgPool2D()(x)

    if include_classification:
        outputs = layers.Dense(nclasses, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    if verbose:
        print(model.summary())

    return model

if __name__ == '__main__':

    audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 10), (1, 10)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'patch_size': 2,
        'spectrogram_dim': (64, 500, 3),
        'verbose': True
    }

    construct_convmixer_model(include_classification=True, nclasses=10, **audio_network_settings)
