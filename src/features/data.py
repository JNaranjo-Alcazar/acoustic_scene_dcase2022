from fileinput import filename
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
  
def get_data(annotations_file,test):
    if test !=True:
        csv = pd.read_csv(annotations_file,sep="\t")
        scene = csv['scene_label'].tolist()
        return scene,csv
    else:
        csv = pd.read_csv(annotations_file,sep="\t")
        filename = csv['filename'].tolist()
        return csv,filename

def get_audio_sample_label(data,label_encoder,onehot_encoder):
    label = label_encoder.transform([data]) #label list
    integer_encoded =label.reshape(len(label), 1)
    onehot_encoded = onehot_encoder.transform(integer_encoded)
    return label,onehot_encoded

def get_audio_dir(audio_dir,annotations_file,index):
    path = os.path.join(audio_dir,annotations_file.iloc[index,0])
    return path
