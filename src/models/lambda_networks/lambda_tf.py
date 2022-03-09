'''
https://github.com/lucidrains/lambda-networks
'''

import tensorflow as tf
from lambda_networks.tfkeras import LambdaLayer

from tensorflow.keras.layers import (Conv2D, MaxPool2D, Dropout, Input, BatchNormalization,
                                    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, ELU)
from tensorflow.keras.models import Model

def construct_lambda_network(include_classification=True, nclasses=10, **parameters):

	spectrogram_dim = parameters['spectrogram_dim']
	filters = parameters['nfilters']
	top_flatten = parameters['top_flatten']
	dropout = parameters['dropout']
	pooling = parameters['pooling']
	kernel_size  = parameters['kernel_size']
	verbose = parameters['verbose']

	receptive_field = parameters['receptive_field']
	key_dimension = parameters['key_dimension']
	heads = parameters['heads']
	intra_depth = parameters['intra_depth']

	inp = Input(shape=spectrogram_dim)

	for i in range(0, len(filters)):
		if i == 0:
            # add lamda layer
			x = LambdaLayer(
            	dim_out = filters[i], # channels out
            	r = receptive_field,       # the receptive field for relative positional encoding (23 x 23)
            	dim_k = key_dimension,   # key dimension
            	heads = heads,   # number of heads, for multi-query; values dimension must be divisible by number of heads for multi-head query
            	dim_u = intra_depth     # 'intra-depth' dimension
        	)(inp)
		else:
            # add lamda layer
			x = LambdaLayer(
            	dim_out = filters[i], # channels out
            	r = receptive_field,       # the receptive field for relative positional encoding (23 x 23)
            	dim_k = key_dimension,   # key dimension
            	heads = heads,   # number of heads, for multi-query; values dimension must be divisible by number of heads for multi-head query
            	dim_u = intra_depth     # 'intra-depth' dimension
        	)(x)
        
		x = BatchNormalization()(x)
		x = ELU()(x)
 
		x = MaxPool2D(pool_size=pooling[i])(x)
		x = Dropout(rate=dropout[i])(x)

	if top_flatten == 'avg':
		x = GlobalAveragePooling2D()(x)
	elif top_flatten == 'max':
		x = GlobalMaxPooling2D()(x)

	x = Dense(units=nclasses, activation='softmax', name='pred_layer')(x)

	model = Model(inputs=inp, outputs=x)

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
        'spectrogram_dim': (64, 500, 3),
        'receptive_field': 23,
        'key_dimension':16,
        'heads': 8,
        'intra_depth': 1,
        'verbose': True
    }

	construct_lambda_network(include_classification=True, **audio_network_settings)
