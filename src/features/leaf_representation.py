import librosa
import numpy as np
import leaf_audio.frontend as frontend

leaf = frontend.Leaf()

def leaf_audio():
    pathAudio = "/acoustic_scene_dcase2022/data/test"
    files = librosa.util.find_files(pathAudio,ext=['wav']) #nos permite obtener cada uno de los archivos
    files = np.array(files)

    for y in files:
        data,sr = librosa.load(y, sr=None) #cargarmos los archivos
        data_array = np.array(data)
        audio_data = data_array[np.newaxis,:] #añadimos una dimensión [1,441000]
        leaf_representation = leaf(audio_data) #llamamos a leaf
        leaf_data = np.array(leaf_representation)
        print(leaf_data.shape,leaf_data)
    return(leaf_data)

leaf_audio()    