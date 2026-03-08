import torch
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()