import os
import shutil
import huggingface_hub
import glob
import json
import copy
import logging
import io
from ipywidgets import widgets
from pathlib import Path
from IPython.display import Audio, display
import soundfile
import numpy as np
import torch
from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc
import typer
from rich.console import Console

class HFModels:
    def __init__(self, repo="therealvul/so-vits-svc-4.0", model_dir="hf_vul_models"):
        self.model_repo = huggingface_hub.Repository(local_dir=model_dir, clone_from=repo, skip_lfs_files=True)
        self.repo = repo
        self.model_dir = model_dir

        self.model_folders = os.listdir(model_dir)
        self.model_folders.remove('.git')
        self.model_folders.remove('.gitattributes')

    def list_models(self):
        return self.model_folders
          
    def extract(self, path):
        if path.endswith(".zip"):
            with ZipFile(path, 'r') as zipObj:
               zipObj.extractall(os.path.split(path)[0])
        elif path.endswith(".tar.bz2"):
            tar = tarfile.open(path, "r:bz2")
            tar.extractall(os.path.split(path)[0])
            tar.close()
        elif path.endswith(".tar.gz"):
            tar = tarfile.open(path, "r:gz")
            tar.extractall(os.path.split(path)[0])
            tar.close()
        elif path.endswith(".tar"):
            tar = tarfile.open(path, "r:")
            tar.extractall(os.path.split(path)[0])
            tar.close()
        elif path.endswith(".7z"):
            archive = py7zr.SevenZipFile(path, mode='r')
            archive.extractall(path=os.path.split(path)[0])
            archive.close()
        else:
            raise NotImplementedError(f"{path} extension not implemented.")

    def download_model(self, model_name, target_dir):
        if model_name not in self.model_folders:
            raise Exception(model_name + " not found")

        charpath = os.path.join(self.model_dir, model_name)

        gen_pt = next(x for x in os.listdir(charpath) if x.startswith("G_"))
        cfg = next(x for x in os.listdir(charpath) if x.endswith("json"))

        try:
            clust = next(x for x in os.listdir(charpath) if x.endswith("pt"))
        except StopIteration:
            print(f"Note - no cluster model for {model_name}")
            clust = None

        os.makedirs(target_dir, exist_ok=True)

        gen_dir = huggingface_hub.hf_hub_download(repo_id=self.repo, filename=f"{model_name}/{gen_pt}")
        shutil.move(os.path.realpath(gen_dir), os.path.join(target_dir, gen_pt))

        shutil.copy(os.path.join(charpath, cfg), os.path.join(target_dir, cfg))

        if clust is not None:
            clust_dir = huggingface_hub.hf_hub_download(repo_id=self.repo, filename=f"{model_name}/{clust}")
            shutil.move(os.path.realpath(clust_dir), os.path.join(target_dir, clust))
            clust_out = os.path.join(target_dir, clust)
        else:
            clust_out = None

        return {"config_path": os.path.join(target_dir, cfg), "generator_path": os.path.join(target_dir, gen_pt), "cluster_path": clust_out}


os.chdir('/content/so-vits-svc')

# Instantiate HFModels class
vul_models = HFModels()

# List available models
print(vul_models.list_models())

# Download a specific model
model_name = "Chara"  # Replace with the desired model name
target_dir = os.path.join("models", model_name)
model_files = vul_models.download_model(model_name, target_dir)
print(model_files)

MODELS_DIR = "models"

def get_speakers():
  speakers = []
  for _,dirs,_ in os.walk(MODELS_DIR):
    for folder in dirs:
      cur_speaker = {}
      # Look for G_****.pth
      g = glob.glob(os.path.join(MODELS_DIR,folder,'G_*.pth'))
      if not len(g):
        print("Skipping "+folder+", no G_*.pth")
        continue
      cur_speaker["model_path"] = g[0]
      cur_speaker["model_folder"] = folder

      # Look for *.pt (clustering model)
      clst = glob.glob(os.path.join(MODELS_DIR,folder,'*.pt'))
      if not len(clst):
        print("Note: No clustering model found for "+folder)
        cur_speaker["cluster_path"] = ""
      else:
        cur_speaker["cluster_path"] = clst[0]

      # Look for config.json
      cfg = glob.glob(os.path.join(MODELS_DIR,folder,'*.json'))
      if not len(cfg):
        print("Skipping "+folder+", no config json")
        continue
      cur_speaker["cfg_path"] = cfg[0]
      with open(cur_speaker["cfg_path"]) as f:
        try:
          cfg_json = json.loads(f.read())
        except Exception as e:
          print("Malformed config json in "+folder)
        for name, i in cfg_json["spk"].items():
          cur_speaker["name"] = name
          cur_speaker["id"] = i
          if not name.startswith('.'):
            speakers.append(copy.copy(cur_speaker))

    return sorted(speakers, key=lambda x:x["name"].lower())

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")
existing_files = []
slice_db = -40
wav_format = 'wav'

class InferenceGui():
  def __init__(self):
    self.speakers = get_speakers()
    self.speaker_list = [x["name"] for x in self.speakers]
    self.speaker_box = widgets.Dropdown(
        options = self.speaker_list
    )
    display(self.speaker_box)

    def convert_cb(btn):
      self.convert()
    def clean_cb(btn):
      self.clean()

    self.convert_btn = widgets.Button(description="Convert")
    self.convert_btn.on_click(convert_cb)
    self.clean_btn = widgets.Button(description="Delete all audio files")
    self.clean_btn.on_click(clean_cb)

    self.trans_tx = widgets.IntText(value=0, description='Transpose')
    self.cluster_ratio_tx = widgets.FloatText(value=0.0, 
      description='Clustering Ratio')
    self.noise_scale_tx = widgets.FloatText(value=0.4, 
      description='Noise Scale')
    self.auto_pitch_ck = widgets.Checkbox(value=False, description=
      'Auto pitch f0 (do not use for singing)')

    display(self.trans_tx)
    display(self.cluster_ratio_tx)
    display(self.noise_scale_tx)
    display(self.auto_pitch_ck)
    display(self.convert_btn)
    display(self.clean_btn)

  def convert(self):
    trans = int(self.trans_tx.value)
    speaker = next(x for x in self.speakers if x["name"] == 
          self.speaker_box.value)
    spkpth2 = os.path.join(os.getcwd(),speaker["model_path"])
    print(spkpth2)
    print(os.path.exists(spkpth2))

    svc_model = Svc(speaker["model_path"], speaker["cfg_path"], 
      cluster_model_path=speaker["cluster_path"])
    
    input_filepaths = [f for f in glob.glob('/content/**/*.*', recursive=True)
     if f not in existing_files and 
     any(f.endswith(ex) for ex in ['.wav','.flac','.mp3','.ogg','.opus'])]
    for name in input_filepaths:
      print("Converting "+os.path.split(name)[-1])
      infer_tool.format_wav(name)

      wav_path = str(Path(name).with_suffix('.wav'))
      wav_name = Path(name).stem
      chunks = slicer.cut(wav_path, db_thresh=slice_db)
      audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

      audio = []
      for (slice_tag, data) in audio_data:
          print(f'#=====segment start, '
              f'{round(len(data)/audio_sr, 3)}s======')
          
          length = int(np.ceil(len(data) / audio_sr *
              svc_model.target_sample))
          
          if slice_tag:
              print('jump empty segment')
              _audio = np.zeros(length)
          else:
              # Padding "fix" for noise
              pad_len = int(audio_sr * 0.5)
              data = np.concatenate([np.zeros([pad_len]),
                  data, np.zeros([pad_len])])
              raw_path = io.BytesIO()
              soundfile.write(raw_path, data, audio_sr, format="wav")
              raw_path.seek(0)
              _cluster_ratio = 0.0
              if speaker["cluster_path"] != "":
                _cluster_ratio = float(self.cluster_ratio_tx.value)
              out_audio, out_sr = svc_model.infer(
                  speaker["name"], trans, raw_path,
                  cluster_infer_ratio = _cluster_ratio,
                  auto_predict_f0 = bool(self.auto_pitch_ck.value),
                  noice_scale = float(self.noise_scale_tx.value))
              _audio = out_audio.cpu().numpy()
              pad_len = int(svc_model.target_sample * 0.5)
              _audio = _audio[pad_len:-pad_len]
          audio.extend(list(infer_tool.pad_array(_audio, length)))
          
      res_path = os.path.join('/content/',
          f'{wav_name}_{trans}_key_'
          f'{speaker["name"]}.{wav_format}')
      soundfile.write(res_path, audio, svc_model.target_sample,
          format=wav_format)
      display(Audio(res_path, autoplay=True)) # display audio file
    pass

  def clean(self):
     input_filepaths = [f for f in glob.glob('/content/**/*.*', recursive=True)
     if f not in existing_files and 
     any(f.endswith(ex) for ex in ['.wav','.flac','.mp3','.ogg','.opus'])]
     for f in input_filepaths:
       os.remove(f)


class App:
    def __init__(self):
        self.inference_gui = None

    def main(self, model_name: str = typer.Argument("Chara", help="Name of the model to download and use")):
        # (Initialize the rest of your classes and functions here)
        console.print(f"Using model: {model_name}", style="bold green")
        self.inference_gui = InferenceGui()

        # Download the specified model
        target_dir = os.path.join("models", model_name)
        model_files = vul_models.download_model(model_name, target_dir)
        console.print(f"Model files: {model_files}", style="bold blue")
        
        logging.getLogger('numba').setLevel(logging.WARNING)
        chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")
        existing_files = []
        slice_db = -40
        wav_format = 'wav'
        
if __name__ == "__main__":
    app = App()
    typer.run(app.main)
