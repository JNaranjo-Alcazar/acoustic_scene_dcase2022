'''
Script to train model
'''

import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchsummary import summary

from baseline.baseline_torch_gammatone import Baseline

# just for debugging purposes
def get_dummy_dataset(in_features=44100, n_classes=10, batch_size=1000):
    X = torch.randn(batch_size, in_features)
    Y = torch.empty(batch_size, dtype=torch.long).random_(n_classes)
    
    X_val = torch.randn(batch_size, in_features)
    Y_val = torch.empty(batch_size, dtype=torch.long).random_(n_classes)

    return X,Y, X_val, Y_val

def parse_args():
    
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Argument for differents types of inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train_features', required=True, type=str,
        help="path to train features")
    parser.add_argument(
        '--train_labels', required=True, type=str,
        help="path to train labels")
    parser.add_argument(
        '--validation_features', required=True, type=str,
        help="path to validation features")
    parser.add_argument(
        '--validation_labels', required=True, type=str,
        help="path to validation labels")
    parser.add_argument(
        '--colab', default=False, type=bool, 
        help="path to setup GPU")
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    
    # Read yaml
    # Import model as yaml
    
    audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 2), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 500, 1),
        'verbose': True
    }
    
    opt = parse_args()
    
    print(f"training features path: {opt.train_features}")
    print(f"training labels path: {opt.train_labels}")
    print(f"validation features path: {opt.val_features}")
    print(f"validation labels path: {opt.val_labels}")
    print(f"Set in colab: {opt.colab}")
    
    X = torch.load(opt.train_features)
    Y = torch.load(opt.train_labels)
    X_val = torch.load(opt.val_features)
    Y_val = torch.load(opt.val_labels)

    model = Baseline(10, **audio_network_settings)
    #summary(model, (1, 44100))

    logger = CSVLogger("lightning_logs", name="my_exp_name")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # Train
    if opt.colab is False:
        trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5, 
                            logger=logger, callbacks=[lr_monitor]) #log_every_n_steps=int(n_total_samples/batch_size)
    elif opt.colab:
        trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=500, 
                            logger=logger, callbacks=[lr_monitor], gpus=1)
    #trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=500, gpus=1)

    #X, Y, X_val, Y_val = get_dummy_dataset()
    train_data = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y, dtype=torch.long))
    val_data = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(Y_val, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)


    # Need to prepare acc and on epoch end, pbar -> video
    trainer.fit(model, train_loader, val_loader)

    # Finish training -> prints, logs, etc