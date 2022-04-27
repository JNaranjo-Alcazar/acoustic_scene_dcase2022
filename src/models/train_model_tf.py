'''
Script to train model
'''

import h5py
import argparse
from baseline.baseline_tf import construct_baseline_model

parser = argparse.ArgumentParser()

parser.add_argument("--training_file", type=str, required=True,
                    help="Path to h5 training file")
parser.add_argument("--validation_file", type=str, required=True,
                    help="Path to h5 validation file")
parser.add_argument("--configuration_file", type=str, required=True,
                    help="Path to configuration file")

opt = parser.parse_args()


# Read yaml

# TODO: read settings file

# Import model as yaml

# TODO: Create model

if type == 'baseline':

    model = construct_baseline_model(include_classification=True, **audio_network_settings)

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

# LR
# Reduce
# EarlyStopping

# Train

model.fit(training_features, training_labels, validation_data=(validation_features, validation_labels), callbacks=[])

# Finish training -> prints, logs, etc