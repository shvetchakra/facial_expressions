import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import EmotionClassifier



def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier()

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def predict_fn(input_data, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    
    tensor_image = image
    inputs = tensor_image.unsqueeze(0)
    device = 'cpu'
    inputs = inputs.float()
    inputs.to(device)
    
    model.to(device)
    with torch.no_grad():
        model.eval()
        logps = model.forward(inputs)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

    return top_class.item()
