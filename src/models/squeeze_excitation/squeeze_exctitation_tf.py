'''
Squeeze-Excitation model
https://ieeexplore.ieee.org/abstract/document/9118879
'''

from tensorflow.keras.layers import (Conv2D, Dense, Permute, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                    Reshape, BatchNormalization, ELU, Lambda, Input, MaxPooling2D, Activation,
                                    Dropout, add, multiply, Maximum)
import tensorflow.keras.backend as k
from tensorflow.keras.models import Model
import tensorflow as tf
 
    
from tensorflow.keras.regularizers import l2
regularization = l2(0.0001)
 
def construct_asc_network_csse(include_classification=True, nclasses=10, **parameters):
    """
    Args:
        include_classification (bool): include classification layer
        **parameters (dict): setting use to construct the network presented in
        (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9118879)
    """
    kernel_size = parameters['kernel_size']
    nfilters = parameters['nfilters']
    pooling = parameters['pooling']
    dropout = parameters['dropout']
    top_flatten = parameters['top_flatten']
    ratio = parameters['ratio']
    pre_act = parameters['pre_act']
    spectrogram_dim = parameters['spectrogram_dim']
    verbose = parameters['verbose']
    merge_type = parameters['merge_type']
 
    inp = Input(shape=spectrogram_dim)
 
    for i in range(0, len(nfilters)):
        if i == 0:
            x = conv_standard_post(inp, kernel_size, nfilters[i], ratio, merge_type, pre_act=pre_act)
        else:
            x = conv_standard_post(x, kernel_size, nfilters[i], ratio, merge_type, pre_act=pre_act)
 
        x = MaxPooling2D(pool_size=pooling[i])(x)
        x = Dropout(rate=dropout[i])(x)

    if top_flatten == 'avg':
         x = GlobalAveragePooling2D()(x)
    elif top_flatten == 'max':
         x = GlobalMaxPooling2D()(x)      
 
    if include_classification:
        x = Dense(units=nclasses, activation='softmax', name='SP_Pred')(x)
    
    model = Model(inputs=inp, outputs=x)
 
    if verbose:
        print(model.summary())
 
    return model
 
 
def conv_standard_post(inp, kernel_size, nfilters, ratio, merge_type, pre_act=False):
    """
    Block presented in (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9118879)
    Args:
        inp (tensor): input to the block
        nfilters (int): number of filters of a specific block
        ratio (int): ratio used in the channel excitation
        pre_act (bool): presented in this work, use a pre-activation residual block
    Returns:
    """
    x1 = inp
 
    if pre_act:
 
        x = BatchNormalization()(inp)
        x = ELU()(x)
        x = Conv2D(nfilters, kernel_size, padding='same')(x)
 
        x = BatchNormalization()(x)
        x = Conv2D(nfilters, kernel_size, padding='same')(x)
 
    else:
 
        x = Conv2D(nfilters, kernel_size, padding='same')(inp)
        x = BatchNormalization()(x)
        x = ELU()(x)

        x = Conv2D(nfilters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
 
    # shortcut
    x1 = Conv2D(nfilters, 1, padding='same')(x1)
    x1 = BatchNormalization()(x1)
 
    x = module_addition(x, x1)
 
    x = ELU()(x)
    
    x = channel_spatial_squeeze_excite(x, merge_type=merge_type, ratio=ratio)
 
    x = module_addition(x, x1)
 
    return x
 
 
def channel_spatial_squeeze_excite(input_tensor, merge_type='sum', ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    """
 
    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)
    
    if merge_type == 'sum':
        x = add([cse, sse])
    elif merge_type == 'max':
        x = Maximum()([cse, sse])
    return x
 
 
def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
 
    init = input_tensor
    channel_axis = 1 if k.image_data_format() == "channels_first" else -1
    filters = _tensor_shape(init)[channel_axis]
    se_shape = (1, 1, filters)
 
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
 
    if k.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
 
    x = multiply([init, se])
    return x
 
 
def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor (): input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks]
        (https://arxiv.org/abs/1803.02579)
    """
 
    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)
 
    x = multiply([input_tensor, se])
    return x
 
 
def module_addition(inp1, inp2):
    """
    Module of addition of two tensors with same H and W, but can have different channels
    If number of channels of the second tensor is the half of the other, this dimension is repeated
    Args:
        inp1 (tensor): one branch of the addition module
        inp2 (tensor): other branch of the addition module
    Returns:
    """
    if k.int_shape(inp1)[3] != k.int_shape(inp2)[3]:
        x = add(
            [inp1, Lambda(lambda y: k.repeat_elements(y, rep=int(k.int_shape(inp1)[3] // k.int_shape(inp2)[3]),
                                                      axis=3))(inp2)])
    else:
        x = add([inp1, inp2])
 
    return x
 
 
def _tensor_shape(tensor):
    """
    Obtain shape in order to use channel excitation
    Args:
        tensor (tensor): input tensor
    Returns:
    """
    return tensor.get_shape()


if __name__ == '__main__':

	audio_network_settings = {
	    'kernel_size': 3,
	    'nfilters': (40, 40),
	    'pooling': [(1, 10), (1, 10)],
	    'dropout': [0.3, 0.3],
	    'top_flatten': 'avg',
	    'ratio': 2,
	    'pre_act': False,
	    'spectrogram_dim': (64, 500, 1),
        'merge_type': 'max',
	    'verbose': True
	}

	audio_model = construct_asc_network_csse(include_classification=True, **audio_network_settings)