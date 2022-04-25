from signal import signal
import numpy as np
from rsa import sign
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import os
from torch.utils.data import Dataset
from src.features.data import *
from src.features.build_features import *

ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/meta.csv"
AUDIO_DIR = "/app/dcasedata"

class DCASEDataset(Dataset):

    def __init__(self,annotations_file,audio_dir):
        self.annotations, self.scene = get_data(annotations_file)
        self.audio_dir = audio_dir
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.scene)
        labels = self.label_encoder.transform(self.scene)
        integer_encoded =labels.reshape(len(labels), 1)
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(integer_encoded)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = get_audio_dir(self.audio_dir,self.annotations,index)
        label, onehot_encoded = get_audio_sample_label(self.scene[index],self.label_encoder,self.onehot_encoder)
        signal,sr = load_audio(audio_sample_path)
        gamma_spec = gammatone(signal)
        leaf_spec = leaf_audio(signal)
        mel = mel_spectogram(signal)
        return audio_sample_path,label,onehot_encoded,signal,sr,gamma_spec,leaf_spec,mel


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/meta.csv"
    AUDIO_DIR = "/app/dcasedata"
    dset = DCASEDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(dset)} samples in the dataset.")
    audio_path, label,onehot_encoder,signal_audio,sr,gamma,leaf,mel = dset[0]
    


    


