'''
Script to train model
'''
import yaml
import h5py
import argparse
from baseline.baseline_tf import construct_baseline_model
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument("--training_file", type=str, required=True,default="//app/acoustic_scene_dcase2022/mels.h5",
                    help="Path to h5 training file")
parser.add_argument("--validation_file", type=str, required=True,default="/app/acoustic_scene_dcase2022/mels_validation.h5",
                    help="Path to h5 validation file")
parser.add_argument("--configuration_file", type=str, required=True,default="/app/acoustic_scene_dcase2022/src/models/configuration.yml",
                    help="Path to configuration file")

opt = parser.parse_args()

# Read yaml

with open ("/app/acoustic_scene_dcase2022/src/models/configuration.yml") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


# TODO: Create model

if type == 'baseline':

    model = construct_baseline_model(include_classification=True, **config)

# Read h5

hf_training = h5py.File(opt.training_file, 'r')
training_features = hf_training['features'][:]
training_labels = hf_training['labels'][:]
print(f'Training features shape: {training_features.shape}')
print(f'Training labels shape: {training_labels.shape}')

hf_val = h5py.File(opt.validation_file, 'r')
validation_features = hf_val['features'][:]
validation_labels = hf_val['labels'][:]
print(f'Training features shape: {validation_features.shape}')
print(f'Training labels shape: {validation_labels.shape}')

# Callbacks

# Model Checkpoint -> https://keras.io/api/callbacks/model_checkpoint/
# CSV logger -> https://keras.io/api/callbacks/csv_logger/
# Reduce -> https://keras.io/api/callbacks/reduce_lr_on_plateau/
# EarlyStopping -> https://keras.io/api/callbacks/early_stopping/

model_callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=""),
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.CSVLogger("training.log")
]

# Train

model.fit(training_features, training_labels, validation_data=(validation_features, validation_labels), callbakcs=model_callbacks)

# Finish training -> prints, logs, etc
