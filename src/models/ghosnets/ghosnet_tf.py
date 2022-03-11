from tensorflow.keras.layers import (Conv2D, MaxPool2D, Dropout, Input, BatchNormalization,
                                    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, ELU)
from tensorflow.keras.models import Model

from ghosnet_module import GhostModule

class ConstructGhosnetModule(GhostModule):
    def __init__(self, nclasses):
        """Init"""
        super(ConstructGhosnetModule, self).__init__(nclasses)
        self.ratio = 2
        self.dw_kernel = 3
        self.nclasses = nclasses
        

    def build(self, **parameters):

        spectrogram_dim = parameters['spectrogram_dim']
        filters = parameters['nfilters']
        top_flatten = parameters['top_flatten']
        dropout = parameters['dropout']
        pooling = parameters['pooling']
        kernel_size  = parameters['kernel_size']
        verbose = parameters['verbose']
        bottleneck = parameters['bottleneck']

        inp = Input(shape=spectrogram_dim)
        # TODO: check hyperparameters
        for i in range(0, len(filters)):
            if i == 0:
                # _ghost_bottleneck(self, inputs, outputs, kernel, dw_kernel, exp, s, ratio, squeeze, name=None):
                if bottleneck:
                    x = self._ghost_bottleneck(inp, filters[i], kernel_size, self.dw_kernel, 16, 1, self.ratio, False)
                else:
                    # _ghost_module(self, inputs, exp, kernel, dw_kernel, ratio, s=1, padding='SAME',use_bias=False, data_format='channels_last', activation=None):
                    x = self._ghost_module(inp, 16, kernel_size, self.dw_kernel, self.ratio)
            else:
                if bottleneck:
                    x = self._ghost_bottleneck(x, filters[i], kernel_size, self.dw_kernel, 16, 1, self.ratio, False)
                else:
                    x = self._ghost_module(x, 16, kernel_size, self.dw_kernel,  self.ratio)
            
            x = BatchNormalization()(x)
            x = ELU()(x)
    
            x = MaxPool2D(pool_size=pooling[i])(x)
            x = Dropout(rate=dropout[i])(x)

        if top_flatten == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif top_flatten == 'max':
            x = GlobalMaxPooling2D()(x)

        x = Dense(units=self.nclasses, activation='softmax', name='pred_layer')(x)

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
        'bottleneck': False,
        'verbose': True
    }

    constructor = ConstructGhosnetModule(nclasses=10)
    model = constructor.build(**audio_network_settings)