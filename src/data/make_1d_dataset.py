'''
Script for creating 1d dataset
'''

from locale import normalize
import os

import pickle

from numpy import array
#from numpy import argmax
from sklearn.preprocessing import LabelEncoder

import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd

from tqdm import tqdm

def read_csv(path_to_file):
    
    dataframe = pd.read_csv(path_to_file, sep='\t')
    return dataframe

def create_1d_matrix(filenames, prefix, sample_rate, mode):
    
    #resampler = T.Resample(44100, sample_rate)
    
    idx = 0
    if mode == 'train':
        X = torch.empty(len(filenames)-1, sample_rate)
    else:
        X = torch.empty(len(filenames), sample_rate)
        
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        if filename !=  'audio/park-vienna-105-2983-6-s1.wav': # problem reading this specific file
            x, sr = torchaudio.load(os.path.join(prefix, filename), normalize=True)
            #x = resampler(x)
            X[idx] = x
            idx += 1
        
    return X


def real_devices(list_files, device):
    
   match = [s for s in list_files if f"-{device}.wav" in s]
   return match
    
        
def create_labeler_encoder(labels):
    
    labels = array(labels)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    return label_encoder

def create_1d_labels(labels, label_encoder):
    
    return label_encoder.transform(array(labels))

if __name__ == '__main__':

    PATH_TO_TRAIN_CSV = ('/app/data/'
                        'TAU-urban-acoustic-scenes-2022-mobile-development.meta/'
                        'TAU-urban-acoustic-scenes-2022-mobile-development/'
                        'evaluation_setup/fold1_train.csv')
    PATH_TO_VALIDATE_CSV = ('/app/data/'
                        'TAU-urban-acoustic-scenes-2022-mobile-development.meta/'
                        'TAU-urban-acoustic-scenes-2022-mobile-development/'
                        'evaluation_setup/fold1_evaluate.csv')
    
    PREFIX_TO_DATA = ('/app/data/TAU-urban-acoustic-scenes-2022-mobile-development.audio/'
                      'TAU-urban-acoustic-scenes-2022-mobile-development/')
    
    SAMPLE_RATE = 44100
    
    dataframe_train = read_csv(PATH_TO_TRAIN_CSV)
    
    train_files = dataframe_train['filename'].tolist()
    real_devices_a = real_devices(train_files, 'a')
    #real_devices_b = real_devices(train_files, 'b')
    #real_devices_c = real_devices(train_files, 'c')
    #real_devices_a.extend(real_devices_b)
    #real_devices_a.extend(real_devices_c)
    X_train = create_1d_matrix(real_devices_a, PREFIX_TO_DATA, SAMPLE_RATE, mode='real') 
    train_scenes = dataframe_train['scene_label'].tolist()
    
    idx_a = [i for i, j in enumerate(train_files) if '-a.wav' in j]
    #idx_b = [i for i, j in enumerate(train_files) if '-b.wav' in j]
    #idx_c = [i for i, j in enumerate(train_files) if '-c.wav' in j]
    #idx_a.extend(idx_b)
    #idx_a.extend(idx_c)
    train_scenes = list(array(train_scenes)[idx_a])
    #train_scenes.pop(68276)
    label_encoder = create_labeler_encoder(train_scenes)
    Y_train = torch.Tensor(create_1d_labels(train_scenes, label_encoder))
    
    if os.path.isdir("/app/data/1d_data/real") is False:
        os.mkdir("/app/data/1d_data/real")
    torch.save(X_train, '/app/data/1d_data/real/X_train.pt')
    torch.save(Y_train, '/app/data/1d_data/real/Y_train.pt')
    
    del train_files
    del X_train
    del train_scenes
    del Y_train
    
    dataframe_val = read_csv(PATH_TO_VALIDATE_CSV)
    val_files = dataframe_val['filename'].tolist()
    real_devices_a = real_devices(val_files, 'a')
    # real_devices_b = real_devices(val_files, 'b')
    # real_devices_c = real_devices(val_files, 'c')
    # real_devices_a.extend(real_devices_b)
    # real_devices_a.extend(real_devices_c)
    X_val = create_1d_matrix(real_devices_a, PREFIX_TO_DATA, SAMPLE_RATE, mode='val')
    val_scenes = dataframe_val['scene_label'].tolist()
    idx_a = [i for i, j in enumerate(val_files) if '-a.wav' in j]
    # idx_b = [i for i, j in enumerate(val_files) if '-b.wav' in j]
    # idx_c = [i for i, j in enumerate(val_files) if '-c.wav' in j]
    # idx_a.extend(idx_b)
    # idx_a.extend(idx_c)
    val_scenes = list(array(val_scenes)[idx_a])
    Y_val = torch.Tensor(create_1d_labels(val_scenes, label_encoder))
    
    if os.path.isdir("/app/data/real/1d_data") is False:
        os.mkdir("/app/data/real/1d_data")
    
    
    torch.save(X_val, '/app/data/1d_data/real/X_val.pt')
    
    torch.save(Y_val, '/app/data/1d_data/real/Y_val.pt')
    pickle.dump(label_encoder, open('/app/data/1d_data/real/label_encoder.pickle','wb'))