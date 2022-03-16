'''
Script to launch for parameter counting
'''

import argparse

from tensorflow.keras.models import load_model

from utils.nessi import nessi

def arg_parser():
    
    parser = argparse.ArgumentParser(description="arguments for inference", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, type=str, help="path to model")
    parser.add_argument("--framework", required=True, type=str, 
                        choices=["torch", "tflite", "tf"], help="deep learning framework")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    opt = arg_parser()
    
    if opt.framework == "torch":
        # load torch model
        nessi.get_model_size(model, opt.framework, input_size=(1, 44100))
    elif opt.framework == "tflite":
        nessi.get_model_size(opt.model, opt.framework)
    elif opt.framework == "tf":
        model = load_model(opt.model)
        nessi.get_model_size(model, opt.framework)
        

