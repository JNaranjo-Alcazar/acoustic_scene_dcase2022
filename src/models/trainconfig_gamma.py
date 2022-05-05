network_type =  'baseline'
framework = 'tensorflow' # ['tensorflow', 'torch']

audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 4), (1, 2)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 49, 1),
        'verbose': True
    }