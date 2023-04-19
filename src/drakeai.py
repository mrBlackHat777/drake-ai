import os
import json
import glob
import copy
import logging
import shutil
import huggingface_hub
import ipywidgets as widgets
from IPython.display import Audio, display
import typer
from rich.console import Console
# from so_vits_svc.inference import infer_tool

class HFModels:
    def __init__(self, repo = "therealvul/so-vits-svc-4.0", model_dir = "hf_vul_models"):
        self.model_repo = huggingface_hub.Repository(local_dir=model_dir,
            clone_from=repo, skip_lfs_files=True)
        self.repo = repo
        self.model_dir = model_dir

        self.model_folders = os.listdir(model_dir)
        self.model_folders.remove('.git')
        self.model_folders.remove('.gitattributes')

    def list_models(self):
        return self.model_folders

    # Downloads model;
    # copies config to target_dir and moves model to target_dir
    def download_model(self, model_name, target_dir):
        if model_name not in self.model_folders:
            raise Exception(f"{model_name} not found")
        model_dir = self.model_dir
        charpath = os.path.join(model_dir,model_name)

        gen_pt = next(x for x in os.listdir(charpath) if x.startswith("G_"))
        cfg = next(x for x in os.listdir(charpath) if x.endswith("json"))
        try:
            clust = next(x for x in os.listdir(charpath) if x.endswith("pt"))
        except StopIteration as e:
            print(f"Note - no cluster model for {model_name}")
            clust = None

        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        gen_dir = huggingface_hub.hf_hub_download(
            repo_id=self.repo, filename=f"{model_name}/{gen_pt}"
        )

        if clust is not None:
            clust_dir = huggingface_hub.hf_hub_download(
                repo_id=self.repo, filename=f"{model_name}/{clust}"
            )
            shutil.move(os.path.realpath(clust_dir), os.path.join(target_dir, clust))
            clust_out = os.path.join(target_dir, clust)
        else:
            clust_out = None

        shutil.copy(os.path.join(charpath,cfg),os.path.join(target_dir, cfg))
        shutil.move(os.path.realpath(gen_dir), os.path.join(target_dir, gen_pt))

        return {"config_path": os.path.join(target_dir,cfg),
            "generator_path": os.path.join(target_dir,gen_pt),
            "cluster_path": clust_out}

def download_and_prepare_models():
    os.chdir('/content/so-vits-svc')

    # Instantiate HFModels class
    voice_models = HFModels()

    # List available models
    print(voice_models.list_models())

    # Download a specific model
    model_name = "Starlight (singing)"  # Replace with the desired model name
    target_folder = os.path.join("models", model_name)
    model_files = voice_models.download_model(model_name, target_folder)
    print(model_files)

    return {"config_path": os.path.join(target_folder, cfg), "generator_path": os.path.join(target_folder, gen_pt), "cluster_path": clust_out}

def get_speakers():
    speakers = []
    MODELS_FOLDER = "models"

    for _, dirs, _ in os.walk(MODELS_FOLDER):
        for folder in dirs:
            # Search for G_****.pth
            g = glob.glob(os.path.join(MODELS_FOLDER, folder, 'G_*.pth'))
            if not len(g):
                print(f"Skipping {folder}, no G_*.pth")
                continue
            current_speaker = {"model_path": g[0], "model_folder": folder}
            # Search for *.pt (clustering model)
            clst = glob.glob(os.path.join(MODELS_FOLDER, folder, '*.pt'))
            if not len(clst):
                print(f"Note: No clustering model found for {folder}")
                current_speaker["cluster_path"] = ""
            else:
                current_speaker["cluster_path"] = clst[0]

            # Search for config.json
            cfg = glob.glob(os.path.join(MODELS_FOLDER, folder, '*.json'))
            if not len(cfg):
                print(f"Skipping {folder}, no configuration json file")
                continue
            current_speaker["cfg_path"] = cfg[0]
            with open(current_speaker["cfg_path"]) as f:
                try:
                    cfg_json = json.loads(f.read())
                except Exception as e:
                    print(f"Malformed configuration json file in {folder}")
                for name, i in cfg_json["spk"].items():
                    current_speaker["name"] = name
                    current_speaker["id"] = i
                    if not name.startswith('.'):
                        speakers.append(copy.copy(current_speaker))

    return sorted(speakers, key=lambda x: x["name"].lower())

class InferenceInterface():
    def __init__(self):
        self.speakers = get_speakers()
        self.speaker_list = [x["name"] for x in self.speakers]
        self.speaker_selector = widgets.Dropdown(
            options=self.speaker_list
        )
        display(self.speaker_selector)

        def convert_cb(btn):
            self.convert()
        def clean_cb(btn):
            self.clean()

        self.convert_button = widgets.Button(description="Convert")
        self.convert_button.on_click(convert_cb)
        self.clean_button = widgets.Button(description="Delete all audio files")
        self.clean_button.on_click(clean_cb)

        display(self.convert_button)
        display(self.clean_button)

    def convert(self):
        if not os.path.exists("output"):
            os.makedirs("output")

        selected_speaker = self.speakers[self.speaker_list.index(self.speaker_selector.value)]

        # Read speaker configuration
        with open(selected_speaker["cfg_path"], "r") as f:
            speaker_cfg = json.load(f)

        # Initialize inference tool
        # infer = infer_tool(speaker_cfg, selected_speaker["model_path"], selected_speaker["cluster_path"])

        # Input text to be synthesized
        text = input("Enter the text to be synthesized: ")

        # Perform inference
        output_file = os.path.join("output", selected_speaker["name"] + ".wav")
        # infer.synthesize(text, output_file)
        print("Audio file saved to:", output_file)

    def clean(self):
        if os.path.exists("output"):
            for file in glob.glob("output/*.wav"):
                os.remove(file)
            print("All audio files have been deleted.")
        else:
            print("No audio files found to delete.")

def main(
    model_name: str = typer.Option(
        "Starlight (singing)",
        help="Name of the voice model to download and prepare.",
    ),
    model_dir: str = typer.Option(
        "models",
        help="Path to the directory where the voice models will be stored.",
    ),
):
    console = Console()
    console.print("Downloading and preparing voice models...", style="bold")

    # Instantiate HFModels class
    voice_models = HFModels()

    # List available models
    available_models = voice_models.list_models()
    console.print(f"Available models: {', '.join(available_models)}")

    if model_name not in available_models:
        console.print(f"{model_name} not found. Exiting.", style="bold red")
        return

    # Download a specific model
    target_folder = os.path.join(model_dir, model_name)
    console.print(f"Downloading {model_name}...", style="bold blue")
    model_files = voice_models.download_model(model_name, target_folder)
    console.print(f"Model files: {model_files}", style="bold green")

if __name__ == "__main__":
    typer.run(main)