
import numpy as np
from gammatone.gtgram import gtgram
from torch import le
from load_configuration import HParam
import librosa
import sys
sys.path.append("/app/acoustic_scene_dcase2022/src/leafaudio")
import leaf_audio.frontend as frontend
import tensorflow as tf

audio_file = "/app/acoustic_scene_dcase2022/LJ001-0001.wav"
hp = HParam("src/features/config.yaml")

def load_audio(audio_file):
    signal,sr = librosa.load(audio_file,sr = hp.audio.sr,mono=True)
    return signal,sr

def gammatone(signal):
    gammatone_spec = gtgram(signal,hp.audio.sr,hp.audio.win_len,hp.audio.hop_len,hp.audio.n_channels,hp.audio.fmin)
    return gammatone_spec

def leaf_audio(signal):
    leaf = frontend.Leaf()
    audio_data = signal[np.newaxis,:]
    leaf_representation = leaf(audio_data)
    return(leaf_representation)

def mel_spectogram(signal):
    return librosa.feature.melspectrogram(signal,hp.audio.sr,win_length=1764,hop_length=882,n_mels=80)



audio,sr = load_audio(audio_file)
gamma = gammatone(audio)
mel = mel_spectogram(audio)
leaf_spec = leaf_audio(audio,leaf)
print(np.shape(audio))
audio_data = audio[np.newaxis,:]
print(np.shape(audio_data))
print("Gammatone shape: ", np.shape(gamma))
print("Mel shape: ", np.shape(mel))
print("Leaf shape: ", np.shape(mel))



