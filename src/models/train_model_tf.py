'''
Script to train model
'''
import yaml
import h5py
import argparse
from baseline.baseline_tf import construct_baseline_model
from squeeze_excitation.squeeze_exctitation_tf import construct_asc_network_csse
from conv_mixer.conv_mixer_tf import construct_convmixer_model
from aug_conv.aug_conv_tf import construct_aug_conv_model
from lambda_model.lambda_tf import construct_lambda_network
from ghosnets.ghosnet_tf import ConstructGhosnetModule 
import tensorflow as tf

import train_config

parser = argparse.ArgumentParser()

parser.add_argument("--training_file", type=str, required=True,default="//app/acoustic_scene_dcase2022/mels.h5",
                    help="Path to h5 training file")
parser.add_argument("--validation_file", type=str, required=True,default="/app/acoustic_scene_dcase2022/mels_validation.h5",
                    help="Path to h5 validation file")
# parser.add_argument("--configuration_file", type=str, required=True,default="/app/acoustic_scene_dcase2022/src/models/configuration.yml",
#                     help="Path to configuration file")

opt = parser.parse_args()

# Read yaml

# with open (opt.configuration_file) as f:
#     config = yaml.safe_load(f)


# TODO: Create model

if train_config.network_type == 'baseline':

    model = construct_baseline_model(include_classification=True, **train_config.audio_network_settings_baseline)

elif train_config.network_type == 'squeeze_excitation':

    model = construct_asc_network_csse(include_classification=True, **train_config.audio_network_settings_scse)

elif train_config.network_type == 'conv_mixer':

    model = construct_convmixer_model(include_classification=True, **train_config.audio_network_settings_convmixer)

elif train_config.network_type == 'aug_conv':

    model = construct_aug_conv_model(include_classification=True, **train_config.audio_network_settings_baseline)

elif train_config.network_type == 'lambda':

    model = construct_lambda_network(include_classification=True, **train_config.audio_network_settings_lambda)
    
elif train_config.network_type == 'ghost':
    
    constructor = ConstructGhosnetModule(nclasses=10)
    model = constructor.build(**train_config.audio_network_settings_ghostnet)

# TODO: ghostnet, mlp_mixer and transformer



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
    tf.keras.callbacks.EarlyStopping(patience=30,monitor='val_categorical_accuracy'),
    tf.keras.callbacks.ModelCheckpoint(filepath=f"/content/drive/MyDrive/TFG/Notebooks/acoustic_scene_dcase2022{train_config.network_type}.h5",monitor='val_categorical_accuracy'), # rellenar path
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15,monitor='val_categorical_accuracy'),
    tf.keras.callbacks.CSVLogger(f"/content/drive/MyDrive/TFG/Notebooks/acoustic_scene_dcase2022/training_{train_config.network_type}.log") # rellenar path
]

# Train
model.compile("adam",loss="categorical_crossentropy",metrics=['categorical_accuracy'])
model.fit(training_features, training_labels, epochs = 500,validation_data=(validation_features, validation_labels), callbacks=model_callbacks)

# Finish training -> prints, logs, etc