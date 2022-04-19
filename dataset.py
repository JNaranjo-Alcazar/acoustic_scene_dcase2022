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
AUDIO_DIR = None

class DCASEDataset(Dataset):

    def __init__(self,annotations_file,audio_dir):
        self.annotations, self.scene = get_data(annotations_file)
        self.audio_dir = audio_dir
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = get_audio_dir(self.audio_dir,self.annotations,index)
        label, onehot_encoded = get_audio_sample_label(self.scene)
        signal,sr = load_audio(audio_sample_path)
        gamma_spec = gammatone(signal)
        leaf_spec = leaf_audio(signal)
        mel = mel_spectogram(signal)

        return audio_sample_path,label,onehot_encoded,signal,sr,gamma_spec,leaf_spec,mel

if __name__ == "__main__":
    dataset = DCASEDataset(ANNOTATIONS_FILE,AUDIO_DIR)
        


    


