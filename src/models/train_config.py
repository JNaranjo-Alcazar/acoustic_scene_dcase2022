network_type =  'baseline'
framework = 'tensorflow' # ['tensorflow', 'torch']

audio_network_settings = {
        'kernel_size': 3,
        'nfilters': (40, 40),
        'pooling': [(1, 10), (1, 10)],
        'dropout': [0.3, 0.3],
        'top_flatten': 'avg',
        'spectrogram_dim': (64, 500, 1),
        'verbose': True
    }