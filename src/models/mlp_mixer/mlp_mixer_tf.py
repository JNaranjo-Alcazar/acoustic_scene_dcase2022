'''
MLP mixer
'''

# https://github.com/Benjamin-Etheredge/mlp-mixer-keras
# Faltaría ver si se puede modificar un poco más, pero como versión inicial, utilizamos este paquete Pip

if __name__ == '__main__':

	audio_network_settings = {
		'spectrogram_dim': (64, 500, 3)
	}

	n_classes = 10

	model = MlpMixerModel(input_shape=audio_network_settings['spectrogram_dim'],
                      num_classes=n_classes, 
                      num_blocks=4, 
                      patch_size=8,
                      hidden_dim=32, 
                      tokens_mlp_dim=64,
                      channels_mlp_dim=128,
                      use_softmax=True)