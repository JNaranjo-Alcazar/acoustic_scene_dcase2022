
import numpy as np
from gammatone.gtgram import gtgram
from load_configuration import HParam
import librosa
from leafaudio.leaf_audio.frontend import frontend
import tensorflow as tf

audio_file = "/app/acoustic_scene_dcase2022/LJ001-0001.wav"
hp = HParam("src/features/config.yaml")
leaf = frontend.Leaf()

def load_audio(audio_file):
    signal,sr = librosa.load(audio_file,sr = hp.audio.sr,mono=True)
    return signal,sr

def gammatone(signal):
    gammatone_spec = gtgram(signal,hp.audio.sr,hp.audio.win_len,hp.audio.hop_len,hp.audio.n_channels,hp.audio.fmin)
    return gammatone_spec

def leaf_audio(signal,leaf):
    audio_data = signal[np.newaxis,:]
    leaf_representation = leaf(audio_data)
    return(leaf_representation)

audio,sr = load_audio(audio_file)
print(audio)
print(np.shape(audio))
audio_data = audio[np.newaxis,:]
print(np.shape(audio_data))