network_type =  'squeeze_excitation'
framework = 'tensorflow' # ['tensorflow', 'torch']

audio_network_settings_baseline = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 4), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 51, 1),
        'verbose': True
}
    
audio_network_settings_scse = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 4), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'ratio': 2,
        'pre_act': False,
        'spectrogram_dim': (64, 51, 1),
        'merge_type': 'sum',  #['max']
        'verbose': True
}

audio_network_settings_convmixer = {
        'kernel_size': 3,
        'nfilters': (32, 64, 128, 256),
        'pooling': [(1, 2), (1, 2), (1, 2), (1, 1)],
        'dropout': [0.3, 0.3, 0.3, 0.3],
        'top_flatten': 'avg',
        'patch_size': 3,
        'spectrogram_dim': (64, 51, 1),
        'verbose': True
    }

audio_network_settings_lambda = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 4), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 51, 1),
        'receptive_field': 23,
        'key_dimension':16,
        'heads': 8,
        'intra_depth': 1,
        'verbose': True
    }
