'''
Implement baseline but in Torch
'''


import torch
from torch.nn import (Linear, ReLU, ELU, 
                     Conv2d, MaxPool2d, Module, ModuleList, 
                     BatchNorm2d, Dropout)
import torch.nn.functional as F

from nnAudio import Spectrogram

from torchsummary import summary

class Baseline(Module):
    def __init__(self, nclasses, **audio_network_settings):
        super(Baseline, self).__init__()

        self.filters = audio_network_settings['nfilters']
        self.dropout = audio_network_settings['dropout']
        self.pooling = audio_network_settings['pooling']
        self.kernel_size  = audio_network_settings['kernel_size']
        self.verbose = audio_network_settings['verbose']
        self.top_flatten = audio_network_settings['top_flatten']
        
        self.conv_layers = ModuleList()
        # Adding Mel trainable layer
        self.conv_layers.append(Spectrogram.MelSpectrogram(n_fft=512, n_mels=64, trainable_mel=True, trainable_STFT=True))

        for idx, nfilter in enumerate(self.filters):
            if idx == 0:
                self.conv_layers.append(Conv2d(1, nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))
            else:
                self.conv_layers.append(Conv2d(self.filters[idx-1], nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))

            self.conv_layers.append(BatchNorm2d(nfilter))
            self.conv_layers.append(ELU(inplace=False))

            self.conv_layers.append(Conv2d(nfilter, nfilter, kernel_size=self.kernel_size, 
                                        stride=1, padding=1))
            self.conv_layers.append(BatchNorm2d(nfilter))
            self.conv_layers.append(ELU(inplace=False))

            self.conv_layers.append(MaxPool2d(kernel_size=self.pooling[idx], stride=self.pooling[idx]))
            self.conv_layers.append(Dropout(p=self.dropout[idx], inplace=False))

        self.classifier = Linear(nfilter, nclasses)

    def forward(self, x):

        i = 0
        for layer in self.conv_layers:
            if i == 1:
                x = layer(x.unsqueeze(1))
            else:
                x = layer(x)
            i += 1
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
        #'spectrogram_dim': (64, 500, 1),
        'verbose': True
    }

    model = Baseline(10, **audio_network_settings)

    print(model)
    #x = model.forward(torch.zeros(1, 441000))
    summary(model, (1, 441000))