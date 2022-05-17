from torch.utils.data import Dataset
from src.features.data import *
from src.features.build_features import *
import numpy as np
import h5py as f
from tqdm import tqdm
import pickle


class DCASEDatasetValidation(Dataset):

    def __init__(self,annotations_file,audio_dir,model):
        self.annotations,self.filename = get_data(annotations_file,test=True)
        self.audio_dir = audio_dir
        self.interpreter = load(model,tflite = True)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = get_audio_dir(self.audio_dir,self.annotations,index)
        signal,sr = load_audio(audio_sample_path)
        signal = resample_if_necessary(signal,sr)
        #gamma_spec = gammatone(signal)
        #leaf_spec = leaf_audio(signal)
        mel = mel_spectogram(signal)
        return audio_sample_path,signal,sr,mel
    
if __name__ == "__main__":
    MODEL = "/app/acoustic_scene_dcase2022/baseline_conv_sep_4848.tflite"
    ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/fold1_test.csv"
    AUDIO_DIR = "/app/dcasedata"
    
    dset = DCASEDatasetValidation(ANNOTATIONS_FILE, AUDIO_DIR,MODEL)
    
    input_details = dset.interpreter.get_input_details()[0]
    output_details = dset.interpreter.get_output_details()[0]

    mel = np.zeros((len(dset),64,51))
    predictions = np.zeros(len(dset))
    
    for i in tqdm(range(100)):
        _,_,_,mel[i,:,:] = dset[i]
        mel_expand=np.expand_dims(mel[i],axis=2)
        mel_expand=np.expand_dims(mel_expand,axis=0)
        
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            mel_expand = mel_expand / input_scale + input_zero_point
            
        mel_expand = mel_expand.astype(input_details["dtype"])
        dset.interpreter.set_tensor(input_details['index'], mel_expand)
        dset.interpreter.invoke()
        output_data = dset.interpreter.get_tensor(output_details['index'])
        predictions[i] = output_data.argmax()

