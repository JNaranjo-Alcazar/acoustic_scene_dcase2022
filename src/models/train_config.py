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