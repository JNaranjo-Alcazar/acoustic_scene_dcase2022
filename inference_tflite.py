from torch.utils.data import Dataset
from src.features.data import *
from src.features.build_features import *
import numpy as np
import h5py as f
from tqdm import tqdm
import pickle
from src.features.spec_normalization import normalization, StatsRecorder

class DCASEDatasetValidation(Dataset):

    def __init__(self,annotations_file,audio_dir,model):
        self.annotations,self.filename = get_data(annotations_file,test=True)
        self.audio_dir = audio_dir
        self.interpreter = load(model,tflite = True)
        self.global_stats = StatsRecorder()
        
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
        
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            mel_expand = mel_expand / input_scale + input_zero_point
            
        mel_expand = mel_expand.astype(input_details["dtype"])
        dset.interpreter.set_tensor(input_details['index'], mel_expand)
        dset.interpreter.invoke()
        output_details = dset.interpreter.get_output_details()[0]
        output_data = dset.interpreter.get_tensor(output_details['index'])
        scale, zero_point= output_details['quantization']
        tflite_output=output_data.astype(np.float32)
        tflite_output= (tflite_output- zero_point)* scale
        inverted_prediction = label_encoder.inverse_transform([np.argmax(tflite_output)])
        scene_label[i]=inverted_prediction[0]
        airport[i] = tflite_output[0,0]
        bus[i] = tflite_output[0,1]
        metro[i]=tflite_output[0,2]
        metro_station[i]=tflite_output[0,3]
        park[i]=tflite_output[0,4]
        public_square[i]=tflite_output[0,5]
        shopping_mall[i]=tflite_output[0,6]
        street_pedestrian[i]=tflite_output[0,7]
        street_traffic[i]=tflite_output[0,8]
        tram[i]=tflite_output[0,9]

    for i in tqdm(range(len(file_name))):
        file_name[i] = file_name[i].split("/")[1]
    
df = pd.DataFrame(list(zip(file_name,scene_label,airport,bus,metro,metro_station,park,
                           public_square,shopping_mall,street_pedestrian,street_traffic,tram)),
                  columns=['filename','scene_label','airport','bus','metro','metro_station','park',
                           'public_square','shopping_mall','street_pedestrian','street_traffic','tram'])
pd.DataFrame.to_csv(df,'/app/acoustic_scene_dcase2022/inference_lite.csv')    