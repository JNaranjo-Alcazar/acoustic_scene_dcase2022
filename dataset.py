import imp
from multiprocessing.spawn import import_main_path
from signal import signal
from matplotlib.pyplot import close
import numpy as np
from rsa import sign
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import os
from torch.utils.data import Dataset
from src.features.data import *
from src.features.build_features import *
import h5py as f
from tqdm import tqdm

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
        signal = resample_if_necessary(signal,sr)
       # gamma_spec = gammatone(signal)
        #leaf_spec = leaf_audio(signal)
        mel = mel_spectogram(signal)
        return audio_sample_path,label,onehot_encoded,signal,sr,mel


if __name__ == "__main__":

    ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/fold1_train.csv"
    AUDIO_DIR = "/app/dcasedata"
    
    dset = DCASEDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    
    print(f"There are {len(dset)} samples in the dataset.")
    
    mel = np.zeros((len(dset),64,51))
    leaf_spec = np.zeros((len(dset),1,50,64))
    gamma_spec = np.zeros((len(dset),64,49))
    onehot_encoder  = np.zeros((len(dset),10))
    
    for i in tqdm(range(len(dset))):
        _,_,onehot_encoder[i],_,_,mel[i,:,:] = dset[i]

    mel_expand=np.expand_dims(mel,axis=3)
    leaf_spec_expand = np.moveaxis(leaf_spec,1,-1)
    gamma_spec_expand=np.expand_dims(gamma_spec,axis=3)

    hf = f.File("mels.h5","w")
    hf.create_dataset("features",data=mel_expand)
    hf.create_dataset("labels",data=onehot_encoder)
    hf.close()

    # hf = f.File("leafs.h5","w")
    # hf.create_dataset("features",data=leaf_spec_expand)
    # hf.create_dataset("labels",data=onehot_encoder)
    # hf.close()

    # hf = f.File("gammas.h5","w")
    # hf.create_dataset("features",data=gamma_spec_expand)
    # hf.create_dataset("labels",data=onehot_encoder)
    # hf.close()
    
    
    