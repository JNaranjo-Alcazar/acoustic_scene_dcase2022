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
        self.model = load(model,tflite=False)
    
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
    MODEL = "/app/acoustic_scene_dcase2022/baseline_conv_sep_4848.h5"
    ANNOTATIONS_FILE = "/app/acoustic_scene_dcase2022/fold1_test.csv"
    AUDIO_DIR = "/app/dcasedata"

    dset = DCASEDatasetValidation(ANNOTATIONS_FILE, AUDIO_DIR,MODEL)

    print(f"There are {len(dset)} samples in the dataset.")

    mel = np.zeros((len(dset),64,51))
    leaf_spec = np.zeros((len(dset),1,50,64))
    gamma_spec = np.zeros((len(dset),64,49))
    onehot_encoder  = np.zeros((len(dset),10))
    file_name = dset.filename
    scene_label = [None] * len(dset)
    airport = [None] * len(dset)
    bus = [None] * len(dset) 
    metro = [None] * len(dset)
    metro_station = [None] * len(dset)
    park = [None] * len(dset)
    public_square = [None] * len(dset)
    shopping_mall = [None] * len(dset)
    street_pedestrian = [None] * len(dset)
    street_traffic = [None] * len(dset)
    tram = [None] * len(dset)
    
    
    file_label  = open("/app/acoustic_scene_dcase2022/label_encoder.pickle","rb")
    label_encoder = pickle.load(file_label)
    file_label.close()

    for i in tqdm(range(100)):
        _,_,_,mel[i,:,:] = dset[i]
        mel_expand=np.expand_dims(mel[i],axis=2)
        mel_expand=np.expand_dims(mel_expand,axis=0)
        prediction = dset.model.predict(mel_expand)
        inverted_prediction = label_encoder.inverse_transform([np.argmax(prediction)])
        scene_label[i]=inverted_prediction[0]
        airport[i] = prediction[0,0]
        bus[i] = prediction[0,1]
        metro[i]=prediction[0,2]
        metro_station[i]=prediction[0,3]
        park[i]=prediction[0,4]
        public_square[i]=prediction[0,5]
        shopping_mall[i]=prediction[0,6]
        street_pedestrian[i]=prediction[0,7]
        street_traffic[i]=prediction[0,8]
        tram[i]=prediction[0,9]
    
    for i in tqdm(range(len(file_name))):
        file_name[i] = file_name[i].split("/")[1]
        
df = pd.DataFrame(list(zip(file_name,scene_label,airport,bus,metro,metro_station,park,
                           public_square,shopping_mall,street_pedestrian,street_traffic,tram)),
                  columns=['filename','scene_label','airport','bus','metro','metro_station','park',
                           'public_square','shopping_mall','street_pedestrian','street_traffic','tram'])
pd.DataFrame.to_csv(df,'/app/acoustic_scene_dcase2022/inference.csv')
                    