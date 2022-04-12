from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import os
#import torchaudio

ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/meta.csv"
AUDIO_DIR = None

class DcaseDataset(Dataset):

    def __init__(self,annotations_file,audio_dir):
        self.annotations = pd.read_csv(annotations_file) 
        self.audio_dir = audio_dir


    def __len__(self):
        return len(self.annotations)        

    def __get_ittem__(self,index):
        audio_sample_path =  self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        values = self.annotations.iloc[index,1]
        return label, values


    def _get_audio_sample_path(self,index):
        path = os.path.join(self.audio_dir,self.annotations.iloc[index,0])
        return path

    def get_audio_sample_label(values):
        label_encoder = LabelEncoder()
        label = label_encoder.fit_transform(values)
        return label


    


