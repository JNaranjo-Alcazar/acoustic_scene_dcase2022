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
    filters = parameters['filters']
    top_flatten = parameters['top_flatten']
    verbose = parameters['verbose']
    patch_size = parameters['patch_size']

    inputs = keras.Input(shape=spectrogram_dim)
    #x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters[0], patch_size)

    # ConvMixer blocks.
    for _ in range(len(filters)):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    if top_flatten == 'avg':
         x = layers.GlobalAveragePooling2D()(x)
    elif top_flatten == 'max':
         x = layers.GlobalMaxPooling2D()(x)
    #x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    if verbose:
        print(model.summary())

    return keras.Model(inputs, outputs)

if __name__ == '__main__':

	audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'patch_size': 2,
        #'pooling': [(1, 10), (1, 10)],
        #'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 500, 3),
        'verbose': True
    }

    audio_model = construct_baseline_model(include_classification=True, **audio_network_settings)
