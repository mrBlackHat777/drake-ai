import os
import shutil
import huggingface_hub
import json
import torch
import numpy as np

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU not available")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
