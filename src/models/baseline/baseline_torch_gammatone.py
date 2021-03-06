'''
Implement baseline with trainable Gammatone in Torch
'''


import torch
from torch.nn import (Linear, ReLU, ELU, 
                     Conv2d, MaxPool2d, Module, ModuleList, 
                     BatchNorm2d, Dropout, CrossEntropyLoss)
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
import torchmetrics

from nnAudio.features import gammatone

from torchsummary import summary

class Baseline(pl.LightningModule):
    def __init__(self, nclasses, **audio_network_settings):
        super(Baseline, self).__init__()

        self.filters = audio_network_settings['nfilters']
        self.dropout = audio_network_settings['dropout']
        self.pooling = audio_network_settings['pooling']
        self.kernel_size  = audio_network_settings['kernel_size']
        self.verbose = audio_network_settings['verbose']
        self.top_flatten = audio_network_settings['top_flatten']
        
        self.valid_acc = torchmetrics.Accuracy()
        #self.loss = CrossEntropyLoss()
        
        self.conv_layers = ModuleList()
        # Adding Mel trainable layer
        self.conv_layers.append(gammatone.Gammatonegram(sr=16000, n_fft=640, n_bins=64, hop_length=320, window='hann', 
                                                        center=True, pad_mode='reflect', power=2.0, 
                                                        htk=False, fmin=20.0, fmax=None, norm=1, 
                                                        trainable_bins=False, trainable_STFT=False))

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
        out = self.classifier(x)

        return out
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75), 
                     "monitor": "val_acc_epoch"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        # forward, logits
        logits = self(x)
        #acc = accuracy(logits, y)
        J = F.cross_entropy(logits, y)
        #self.log("accuracy",  acc, prog_bar=True)
        self.log('loss', J, on_step=False, on_epoch=True)
        #return {"loss": J, "acc": acc}
        return J
    
    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.tensor([x["loss"] for x in outputs]).mean()
    #     #avg_accuracy = torch.tensor([x["acc"] for x in outputs]).mean()
    #     #self.log("acc_epoch", avg_accuracy, prog_bar=True)
        
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        a, y_hat = torch.max(logits, dim=1)
        valid_acc = self.valid_acc(y_hat, y)
        #self.log('valid_loss', loss, on_step=False, on_epoch=True)
        #self.log('valid_acc', valid_acc, on_step=False, on_epoch=True)
        
        return {'valid_loss': loss, 'valid_acc': valid_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["valid_acc"] for x in outputs]).mean()
        self.log("val_acc_epoch", avg_accuracy, prog_bar=True)
        return {"val_loss": avg_loss}


if __name__ == '__main__':

    audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 2), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        #'spectrogram_dim': (64, 500, 1),
        'verbose': True
    }

    model = Baseline(10, **audio_network_settings)

    print(model)
    #x = model.forward(torch.zeros(1, 441000))
    summary(model, (1, 16000))