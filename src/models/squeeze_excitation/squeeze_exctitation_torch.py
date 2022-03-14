'''
Squeeze-Excitation model
'''

import torch
from torch.nn import (Linear, ReLU, ELU, 
                     Conv2d, MaxPool2d, Module, ModuleList, 
                     BatchNorm2d, Dropout)
import torch.nn.functional as F

from squeeze_exctitation_module_torch import ChannelSpatialSELayer

from torchsummary import summary

class ChannelSpatialSENetwork(Module):
    def __init__(self, nclasses, **audio_network_settings):
        super(ChannelSpatialSENetwork, self).__init__()

        self.filters = audio_network_settings['nfilters']
        self.dropout = audio_network_settings['dropout']
        self.pooling = audio_network_settings['pooling']
        self.kernel_size  = audio_network_settings['kernel_size']
        self.verbose = audio_network_settings['verbose']
        self.top_flatten = audio_network_settings['top_flatten']
        self.ratio = audio_network_settings['ratio']
        
        self.conv_layers = ModuleList()
        self.shortcut_layers = ModuleList()

        for idx, nfilter in enumerate(self.filters):
            if idx == 0:
                self.conv_layers.append(Conv2d(1, nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))
                self.shortcut_layers.append(Conv2d(1, nfilter, kernel_size=1, 
                                        stride=1))
                
            else:
                self.conv_layers.append(Conv2d(self.filters[idx-1], nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))
                self.shortcut_layers.append(Conv2d(self.filters[idx-1], nfilter, kernel_size=1, 
                                        stride=1))
            
                
            self.conv_layers.append(BatchNorm2d(nfilter))
            self.conv_layers.append(ELU(inplace=False))
            
            self.conv_layers.append(Conv2d(nfilter, nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))
            self.conv_layers.append(BatchNorm2d(nfilter))
                
            self.conv_layers.append(ChannelSpatialSELayer(nfilter, self.ratio))

            self.conv_layers.append(MaxPool2d(kernel_size=self.pooling[idx], stride=self.pooling[idx]))
            self.conv_layers.append(Dropout(p=self.dropout[idx], inplace=False))

        self.classifier = Linear(nfilter, nclasses)

    def forward(self, x):

        idx = 0
        first_idx = 0
        idx_shortcut = 0
        for layer in self.conv_layers:
            if first_idx == 0:
                x1 = x
                first_idx += 1 # first identity
            x = layer(x)
            if isinstance(layer, ChannelSpatialSELayer):
                x = torch.nn.functional.elu(x1 + x) # second addition after scSE
            if isinstance(layer, torch.nn.Dropout):
                x1 = x # getting new identity
                idx = -1 # init counter for next block
            idx += 1
            if idx == 5: # number of layers before identity addition
                x1 = self.shortcut_layers[idx_shortcut](x1) # getting shorchut 1x1 conv
                x = x1 + x  # addition
                idx_shortcut += 1 # next layer for shorcut
        if self.top_flatten == 'avg':
            x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        out = F.softmax(self.classifier(x))

        return out


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

    model = ChannelSpatialSENetwork(10, **audio_network_settings)

    print(model)
    x = model.forward(torch.zeros(32, 1, 64, 500))
    summary(model, (1, 64, 500))