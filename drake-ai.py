import os
import urllib.request
from zipfile import ZipFile
import tarfile
import py7zr
import subprocess
from time import sleep
import gdown
import huggingface_hub
from ipywidgets import widgets

def extract(path):
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


# Optimized code

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


vul_models = HFModels()
os.chdir('/content/so-vits-svc')
download(["https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best
