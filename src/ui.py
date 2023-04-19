import numpy as np
import soundfile
from inference.infer_tool import Svc
from inference import slicer
from inference import infer_tool
import torch
import os
import glob
import json
import copy
import logging
import io
from ipywidgets import widgets
from pathlib import Path
from IPython.display import Audio, display

from src.download_utils import Downloader

os.chdir('/content/so-vits-svc')


vul_models = Downloader("therealvul/so-vits-svc-4.0",
                        "models").hf_models  # HFModels()


def button_eventhandler(but):
    vul_models.download_model(but.description, f"models/{but.description}")


for model in vul_models.list_models():
    btn = widgets.Button(description=model)
    btn.on_click(button_eventhandler)
    display(btn)

MODELS_DIR = "models"


class InferenceApp:

    def __init__(self):
        self.speakers = self.get_speakers()
        self.speaker_list = [x["name"] for x in self.speakers]
        self.create_widgets()

    def get_speakers(self):
        speakers = []
        for _, dirs, _ in os.walk(MODELS_DIR):
            for folder in dirs:
                # Look for G_****.pth
                g = glob.glob(os.path.join(MODELS_DIR, folder, 'G_*.pth'))
                if not len(g):
                    print(f"Skipping {folder}, no G_*.pth")
                    continue
                cur_speaker = {"model_path": g[0], "model_folder": folder}
                # Look for *.pt (clustering model)
                clst = glob.glob(os.path.join(MODELS_DIR, folder, '*.pt'))
                if not len(clst):
                    print(f"Note: No clustering model found for {folder}")
                    cur_speaker["cluster_path"] = ""
                else:
                    cur_speaker["cluster_path"] = clst[0]

                # Look for config.json
                cfg = glob.glob(os.path.join(MODELS_DIR, folder, '*.json'))
                if not len(cfg):
                    print(f"Skipping {folder}, no config json")
                    continue
                cur_speaker["cfg_path"] = cfg[0]
                with open(cur_speaker["cfg_path"]) as f:
                    try:
                        cfg_json = json.loads(f.read())
                    except Exception as e:
                        print(f"Malformed config json in {folder}")
                    for name, i in cfg_json["spk"].items():
                        cur_speaker["name"] = name
                        cur_speaker["id"] = i
                        if not name.startswith('.'):
                            speakers.append(copy.copy(cur_speaker))

        return sorted(speakers, key=lambda x: x["name"].lower())

    def create_widgets(self):
        self.speaker_box = widgets.Dropdown(options=self.speaker_list)
        display(self.speaker_box)

        self.trans_tx = widgets.IntText(value=0, description='Transpose')
        self.cluster_ratio_tx = widgets.FloatText(
            value=0.0, description='Clustering Ratio')
        self.noise_scale_tx = widgets.FloatText(
            value=0.4, description='Noise Scale')
        self.auto_pitch_ck = widgets.Checkbox(
            value=False, description='Auto pitch f0 (do not use for singing)')

        display(self.trans_tx)
        display(self.cluster_ratio_tx)
        display(self.noise_scale_tx)
        display(self.auto_pitch_ck)

        self.convert_btn = widgets.Button(description="Convert")
        self.convert_btn.on_click(self.convert_cb)
        self.clean_btn = widgets.Button(description="Delete all audio files")
        self.clean_btn.on_click(self.clean_cb)

        display(self.convert_btn)
        display(self.clean_btn)

    def convert_cb(self, btn):
        self.convert()

    def clean_cb(self, btn):
        self.clean()

    def convert(self):
        trans = int(self.trans_tx.value)
        speaker = next(
            x for x in self.speakers if x["name"] == self.speaker_box.value)
        spkpth2 = os.path.join(os.getcwd(), speaker["model_path"])

        svc_model = Svc(speaker["model_path"], speaker["cfg_path"],
                        cluster_model_path=speaker["cluster_path"])

        input_filepaths = [f for f in glob.glob('/content/**/*.*', recursive=True) if f not in self.existing_files and any(f.endswith(ex) for ex in ['.wav', '.flac', '.mp3', '.ogg', '.opus'])]

        for name in input_filepaths:
            print(f"Converting {os.path.split(name)[-1]}")
            infer_tool.format_wav(name)

            wav_path = str(Path(name).with_suffix('.wav'))
            wav_name = Path(name).stem
            chunks = slicer.cut(wav_path, db_thresh=self.slice_db)
            audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, 'f'{round(len(data) / audio_sr, 3)}s======')

                length = int(
                    np.ceil(len(data) / audio_sr * svc_model.target_sample))

                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    pad_len = int(audio_sr * 0.5)
                    data = np.concatenate(
                        [np.zeros([pad_len]), data, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)
                    _cluster_ratio = 0.0
                    if speaker["cluster_path"] != "":
                        _cluster_ratio = float(self.cluster_ratio_tx.value)
                    out_audio, out_sr = svc_model.infer(
                        speaker["name"], trans, raw_path,
                        cluster_infer_ratio=_cluster_ratio,
                        auto_predict_f0=bool(self.auto_pitch_ck.value),
                        noice_scale=float(self.noise_scale_tx.value))
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * 0.5)
                    _audio = _audio[pad_len:-pad_len]
                audio.extend(list(infer_tool.pad_array(_audio, length)))

            res_path = os.path.join('/content/',
                                    f'{wav_name}_{trans}_key_'
                                    f'{speaker["name"]}.{self.wav_format}')
            soundfile.write(res_path, audio,
                            svc_model.target_sample, format=self.wav_format)
            display(Audio(res_path, autoplay=True))  # display audio file

    def clean(self):
        input_filepaths = [f for f in glob.glob('/content/**/*.*', recursive=True) if f not in self.existing_files and any(f.endswith(ex) for ex in ['.wav', '.flac', '.mp3', '.ogg', '.opus'])]
        for f in input_filepaths:
            os.remove(f)


if __name__ == '__main__':
    inference_app = InferenceApp()
