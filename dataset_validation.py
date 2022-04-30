from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from src.features.data import *
from src.features.build_features import *
import numpy as np
import h5py as f
from tqdm import tqdm
import pickle

class DCASEDatasetValidation(Dataset):

    def __init__(self,annotations_file,audio_dir,label_file,onehot_encoder_file):
        self.annotations, self.scene = get_data(annotations_file)
        self.audio_dir = audio_dir
        file_label  = open(label_file,"rb")
        self.label_encoder = pickle.load(file_label)
        file_label.close()
        file_onehot  = open(onehot_encoder_file,"rb")
        self.onehot_encoder = pickle.load(file_onehot)
        file_onehot.close()
 
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = get_audio_dir(self.audio_dir,self.annotations,index)
        label, onehot_encoded = get_audio_sample_label(self.scene[index],self.label_encoder,self.onehot_encoder)
        signal,sr = load_audio(audio_sample_path)
        signal = resample_if_necessary(signal,sr)
        gamma_spec = gammatone(signal)
        #leaf_spec = leaf_audio(signal)
        #mel = mel_spectogram(signal)
        return audio_sample_path,label,onehot_encoded,signal,sr,gamma_spec


if __name__ == "__main__":
    LABEL_FILE = "/app/acoustic_scene_dcase2022/label_encoder.pickel"
    ONEHOT_ENCODER_FILE = "/app/acoustic_scene_dcase2022/onehot_encoder.pickle"
    ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/fold1_evaluate.csv"
    AUDIO_DIR = "/app/dcasedata"

    dset = DCASEDatasetValidation(ANNOTATIONS_FILE, AUDIO_DIR,LABEL_FILE,ONEHOT_ENCODER_FILE)

    print(f"There are {len(dset)} samples in the dataset.")

    mel = np.zeros((len(dset),64,51))
    leaf_spec = np.zeros((len(dset),1,50,64))
    gamma_spec = np.zeros((len(dset),64,49))
    onehot_encoder  = np.zeros((len(dset),10))

    for i in tqdm(range(len(dset))):
        _,_,onehot_encoder[i],_,_,gamma_spec[i,:,:] = dset[i]

    #mel_expand=np.expand_dims(mel,axis=3)
    #leaf_spec_expand = np.moveaxis(leaf_spec,1,-1)
    gamma_spec_expand=np.expand_dims(gamma_spec,axis=3)

    # hf = f.File("mels_validation.h5","w")
    # hf.create_dataset("features",data=mel_expand)
    # hf.create_dataset("labels",data=onehot_encoder)
    # hf.close()

    # with open("onehot_encoder.pickle", "wb") as f_onehot:
    #     pickle.dump(dset.onehot_encoder, f_onehot)
    # f_onehot.close()

    # with open("label_encoder.pickel", "wb") as f_label:
    #     pickle.dump(dset.label_encoder, f_label)
    # f_label.close()

    # hf = f.File("leafs_validation.h5","w")
    # hf.create_dataset("features",data=leaf_spec_expand)
    # hf.create_dataset("labels",data=onehot_encoder)
    # hf.close()

    hf = f.File("gammas.h5","w")
    hf.create_dataset("features",data=gamma_spec_expand)
    hf.create_dataset("labels",data=onehot_encoder)
    hf.close()


