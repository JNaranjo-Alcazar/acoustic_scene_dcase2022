
from statistics import mode
import numpy as np
import sys
sys.path.append("/app/acoustic_scene_dcase2022/src/gammatone")
from gammatone.gtgram import gtgram
sys.path.append("/app/acoustic_scene_dcase2022/src/features")
from load_configuration import HParam
import librosa
sys.path.append("/app/acoustic_scene_dcase2022/src/leafaudio")
import leaf_audio.frontend as frontend
import tensorflow as tf

audio_file = "/app/acoustic_scene_dcase2022/LJ001-0001.wav"
hp = HParam("/app/acoustic_scene_dcase2022/src/features/config.yaml")

def load_audio(audio_file):
    signal,sr = librosa.load(audio_file,sr = hp.audio.sr,mono=True)
    return signal,sr

def resample_if_necessary(signal,sr):
    if sr != hp.audio.sr:
        signal =  librosa.resample(signal,sr,44100)
    return signal

def pad_if_necessary(signal):
    pass

def gammatone(signal):
    gammatone_spec = gtgram(signal,hp.audio.sr,hp.audio.win_len,hp.audio.hop_len,hp.audio.n_channels,hp.audio.fmin)
    return np.flipud(20*np.log10(gammatone_spec))

def leaf_audio(signal):
    leaf = frontend.Leaf(n_filters=64,window_len =float(hp.audio.win_len*1000),
                         window_stride=float(hp.audio.hop_len*1000),sample_rate = hp.audio.sr)
    audio_data = signal[np.newaxis,:]
    leaf_representation = leaf(audio_data)
    return(leaf_representation)

def mel_spectogram(signal):
    mel = librosa.feature.melspectrogram(signal,hp.audio.sr,win_length=int(hp.audio.sr*hp.audio.win_len),
                                          hop_length=int(hp.audio.sr*hp.audio.hop_len),n_mels=hp.audio.n_channels)
    return librosa.power_to_db(mel)

def load(model):
    model = tf.keras.models.load_model(model)
    return model

#audio,sr = load_audio(audio_file)
#print(np.shape(audio))
#gamma = gammatone(audio)
#mel = mel_spectogram(audio)
#leaf_spec = leaf_audio(audio)
#print(np.shape(audio))
#audio_data = audio[np.newaxis,:]
#print(np.shape(audio_data))
#print("Gammatone shape: ", np.shape(gamma))
#print("Mel shape: ", np.shape(mel))
#print("Leaf shape: ", np.shape(mel))




