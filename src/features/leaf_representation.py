import librosa
import numpy as np
import leaf_audio.frontend as frontend

leaf = frontend.Leaf()

def leaf_audio(filename: str, verbose: bool = False) -> np.ndarray:
    """Obtains a LEAF representation from audio file

    Args:
        filename (str): full path to audio
        verbose (bool, optional): shows leaf shape. Defaults to False.

    Returns:
        np.ndarray: 2D LEAF representation
    """
    # pathAudio = "/acoustic_scene_dcase2022/data/test"
    # files = librosa.util.find_files(pathAudio,ext=['wav']) #nos permite obtener cada uno de los archivos
    # files = np.array(files)

    # for y in files:
    data,sr = librosa.load(y, sr=None) #cargarmos los archivos
    data_array = np.array(data)
    audio_data = data_array[np.newaxis,:] #añadimos una dimensión [1,441000]
    leaf_representation = leaf(audio_data) #llamamos a leaf
    leaf_data = np.array(leaf_representation)
    if verbose:
        print(leaf_data.shape,leaf_data)
    return(leaf_data)

if __name__ == '__main__':

    leaf_audio('path/to/wav')    