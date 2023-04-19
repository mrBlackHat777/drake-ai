#!/bin/bash

set -e

echo "Cloning repository..."
git clone https://github.com/effusiveperiscope/so-vits-svc -b eff-4.0
cd so-vits-svc

echo "Installing requirements one-at-a-time to ignore exceptions..."
cat requirements.txt | xargs -n 1 pip install --extra-index-url https://download.pytorch.org/whl/cu116

echo "Installing additional packages..."
pip install praat-parselmouth
pip install ipywidgets
pip install huggingface_hub
pip install pip==23.0.1
pip install fairseq==0.12.2
jupyter nbextension enable --py widgetsnbextension

echo "Installing specific package versions..."
pip install numpy==1.21
pip install --upgrade protobuf==3.9.2

echo "Uninstalling and installing specific TensorFlow version..."
pip uninstall -y tensorflow
pip install tensorflow==2.11.0

echo "Installing specific PyTorch version..."
pip install --extra-index-url https://download.pytorch.org/whl/cu116 -r requirements.txt