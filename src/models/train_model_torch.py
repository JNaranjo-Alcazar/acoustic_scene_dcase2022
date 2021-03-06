'''
Script to train model
'''

import gc

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

    return X, Y, X_val, Y_val


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
    #opt = type('', (), {})()
    #opt.train_features = 'D:\\JAVIER\\datasets\\DCASE2022\\task1\\data\\1d_data\\X_train.pt'
    #opt.train_labels = 'D:\\JAVIER\\datasets\\DCASE2022\\task1\\data\\1d_data\\Y_train.pt'
    #opt.validation_features = 'D:\\JAVIER\\datasets\\DCASE2022\\task1\\data\\1d_data\\X_val.pt'
    #opt.validation_labels = 'D:\\JAVIER\\datasets\\DCASE2022\\task1\\data\\1d_data\\Y_val.pt'
    #opt.colab = True

    print(f"training features path: {opt.train_features}")
    print(f"training labels path: {opt.train_labels}")
    print(f"validation features path: {opt.validation_features}")
    print(f"validation labels path: {opt.validation_labels}")
    print(f"Set in colab: {opt.colab}")

    X = torch.load(opt.train_features)
    Y = torch.load(opt.train_labels)
    Y = Y.long()
    train_data = torch.utils.data.TensorDataset(X, Y)
    del (X)
    gc.collect()
    del (Y)
    gc.collect()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    del (train_data)
    gc.collect()

    X_val = torch.load(opt.validation_features)
    Y_val = torch.load(opt.validation_labels)
    Y_val = Y_val.long()
    val_data = torch.utils.data.TensorDataset(X_val, Y_val)
    del (X_val)
    gc.collect()
    del (Y_val)
    gc.collect()
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
    del (val_data)
    gc.collect()

    model = Baseline(10, **audio_network_settings)
    # summary(model, (1, 44100))

    logger = CSVLogger("lightning_logs", name="my_exp_name")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # Train
    if opt.colab is False:
        trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5,
                             logger=logger, callbacks=[lr_monitor])  # log_every_n_steps=int(n_total_samples/batch_size)
    elif opt.colab:
        trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=500,
                             logger=logger, callbacks=[lr_monitor], reload_dataloaders_every_epoch=True,
                             gpus=1)
    # trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=500, gpus=1)

    # X, Y, X_val, Y_val = get_dummy_dataset()

    # Need to prepare acc and on epoch end, pbar -> video
    trainer.fit(model, train_loader, val_loader)

    # Finish training -> prints, logs, etc
