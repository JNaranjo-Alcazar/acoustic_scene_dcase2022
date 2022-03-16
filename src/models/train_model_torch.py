'''
Script to train model
'''

import numpy as np
import torch

import pytorch_lightning as pl
from torchsummary import summary

from baseline.baseline_torch_gammatone import Baseline

# just for debugging purposes
def get_dummy_dataset(in_features=44100, n_classes=10, batch_size=1000):
    X = torch.randn(batch_size, in_features)
    Y = torch.empty(batch_size, dtype=torch.long).random_(n_classes)
    
    X_val = torch.randn(batch_size, in_features)
    Y_val = torch.empty(batch_size, dtype=torch.long).random_(n_classes)

    return X,Y, X_val, Y_val

# Read yaml

audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 2), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 500, 1),
        'verbose': True
    }

# Import model as yaml

model = Baseline(10, **audio_network_settings)
summary(model, (1, 44100))

# Train
trainer = pl.Trainer()

X, Y, X_val, Y_val = get_dummy_dataset()
train_data = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y, dtype=torch.long))
val_data = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(Y_val, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)


# Need to prepare acc and on epoch end, pbar -> video
trainer.fit(model, train_loader, val_loader)

# Finish training -> prints, logs, etc