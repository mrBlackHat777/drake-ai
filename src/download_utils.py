"""
    Download utils for downloading datasets and models from the internet.
    #@markdown Example URLs:
    #@markdown * Kanye: https://mega.nz/file/P7hWwCoQ#s0OICnRbTpcUjUIS7iQPIlYwBVelZXzm_-1LLPSUd2Y
    #@markdown * Kendrick: https://mega.nz/file/WmBzgSZa#UD-SFhHBv3aw0obTHW2lGc5yeaMnK8qtKU3OjDKMVKk
    #@markdown * Carti (v3): https://mega.nz/file/jnwzEJ4K#erlpUaNQ3VyQIIVaQDYge3Kv4pZtyNfQBWA6hUy6uu8
    #@markdown * Drake: https://mega.nz/file/Sm53wAwI#4PmIrSWDrEP1-pnZb5MJpTcfoHy3OBhBOhn2FVxfyb8
    #@markdown * Juice Wrld: https://mega.nz/file/5w9kGSJA#MQEQi7lBBBJMBa_rQ5mfGtDXnv96-XhhsNx-xc81ta8
    #@markdown * Travis: https://mega.nz/file/q652kCZb#VS9IE0Vr3A3PbynDvmkfantFmz_Iik9i3M9DMWeShoE
    #@markdown * Tyler: https://mega.nz/file/rz5SBBIK#KAhHX8tR-f5yf_aR4dRwF-oE90LpliA4E8v1YFC7ONQ
    #@markdown * AI Hub: https://discord.gg/Aktyxz4jwA
"""

from genericpath import exists
import glob
import os
import re
import tarfile
import urllib.request
import subprocess
from zipfile import ZipFile
from time import sleep
from sys import platform
from tqdm import tqdm
import gdown
import huggingface_hub
import shutil


class Downloader:
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    class HFModels:
        def __init__(self, repo="therealvul/so-vits-svc-4.0", model_dir="hf_vul_models"):
            self.model_repo = huggingface_hub.Repository(local_dir=model_dir,clone_from=repo, skip_lfs_files=True)
            self.repo = repo
            self.model_dir = model_dir

            self.model_folders = os.listdir(model_dir)
            self.model_folders.remove('.git')
            self.model_folders.remove('.gitattributes')

        def list_models(self):
            return self.model_folders

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

    def __init__(self):
        self.hf_models = self.HFModels()

    def extract(self, path):
        """Extracts a file to the same directory."""
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
            import py7zr
            archive = py7zr.SevenZipFile(path, mode='r')
            archive.extractall(path=os.path.split(path)[0])
            archive.close()
        else:
            raise NotImplementedError(f"{path} extension not implemented.")

    def megadown(self, download_link, filename='.', verbose=False):
        """Use megatools binary executable to download files and folders from MEGA.nz ."""
        win64_url = "https://megatools.megous.com/builds/builds/megatools-1.11.1.20230212-win64.zip"
        win32_url = "https://megatools.megous.com/builds/builds/megatools-1.11.1.20230212-win32.zip"
        linux_url = "https://megatools.megous.com/builds/builds/megatools-1.11.1.20230212-linux-x86_64.tar.gz"

        if platform in ["linux", "linux2"]:
            dl_url = linux_url
        elif platform == "darwin":
            raise NotImplementedError('MacOS not supported.')
        elif platform == "win32":
                dl_url = win64_url
        else:
            raise NotImplementedError ('Unknown Operating System.')

        dlname = dl_url.split("/")[-1]
        if dlname.endswith(".zip"):
            binary_folder = dlname[:-4] # remove .zip
        elif dlname.endswith(".tar.gz"):
            binary_folder = dlname[:-7] # remove .tar.gz
        else:
            raise NameError('downloaded megatools has unknown archive file extension!')

        if not os.path.exists(binary_folder):
            print('"megatools" not found. Downloading...')
            if not os.path.exists(dlname):
                urllib.request.urlretrieve(dl_url, dlname)
            assert os.path.exists(dlname), 'failed to download.'
            self.extract(dlname)
            sleep(0.10)
            os.unlink(dlname)
            print("Done!")


        binary_folder = os.path.abspath(binary_folder)
        filename = f' --path "{os.path.abspath(filename)}"' if filename else ""
        wd_old = os.getcwd()
        os.chdir(binary_folder)
        try:
            if platform in ["linux", "linux2"]:
                subprocess.call(f'./megatools dl{filename}{" --debug http" if verbose else ""} {download_link}', shell=True)
            elif platform == "win32":
                subprocess.call(f'megatools.exe dl{filename}{" --debug http" if verbose else ""} {download_link}', shell=True)
        except:
            os.chdir(wd_old) # don't let user stop download without going back to correct directory first
            raise
        os.chdir(wd_old)
        return filename

    def download_url(self, url, filename):
        with self.DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, headers = urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
            print(f"Downloaded to {filename}")

    def request_url_with_progress_bar(self, url, filename):
        self.download_url(url, filename)

    def download(self, urls, dataset='', filenames=None, force_dl=False, username='', password='', auth_needed=False):
        assert filenames is None or len(urls) == len(filenames), f"number of urls does not match filenames. Expected {len(filenames)} urls, containing the files listed below.\n{filenames}"
        assert not auth_needed or (len(username) and len(password)), f"username and password needed for {dataset} Dataset"
        if filenames is None:
            filenames = [None,]*len(urls)
        for i, (url, filename) in enumerate(zip(urls, filenames)):
            print(f"Downloading File from {url}")
            #if filename is None:
            #    filename = url.split("/")[-1]
            if filename and (not force_dl) and exists(filename):
                print(f"{filename} Already Exists, Skipping.")
                continue
            if 'drive.google.com' in url:
                assert 'https://drive.google.com/uc?id=' in url, 'Google Drive links should follow the format "https://drive.google.com/uc?id=1eQAnaoDBGQZldPVk-nzgYzRbcPSmnpv6".\nWhere id=XXXXXXXXXXXXXXXXX is the Google Drive Share ID.'
                gdown.download(url, filename, quiet=False)
            elif 'mega.nz' in url:
                self.megadown(url, filename)
            else:
                #urllib.request.urlretrieve(url, filename=filename) # no progress bar
                self.request_url_with_progress_bar(url, filename) # with progress bar


    def get_next_model(self, model_name):
        models = self.hf_models.list_models()
        models.sort()
        index = models.index(model_name)
        return models[index + 1] if index + 1 < len(models) else None

    def download_model(self, model_name, target_dir):
        if model_name not in self.hf_models.list_models():
            model_name = self.get_next_model(model_name)
            if model_name is None:
                raise Exception("No next model found")
        return self.hf_models.download_model(model_name, target_dir)

    def download_models(self, model_names, target_dir):
        for model_name in model_names:
            self.download_model(model_name, target_dir)

    def download_all_models(self, target_dir):
        self.download_models(self.hf_models.list_models(), target_dir)

    def download_contentVec(self, target_dir):
        self.download(["https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt"], filenames=["hubert/checkpoint_best_legacy_500.pt"])

    def download_and_extract_zip(self, model_url):
        if "huggingface.co" in model_url.lower():
            self.download([re.sub(r"/blob/", "/resolve/", model_url)], 
                          filenames=[os.path.join(os.getcwd(), model_url.split("/")[-1])])
        else:
            self.download([model_url])

        os.makedirs('models', exist_ok=True)
        model_zip_paths = glob.glob('/content/**/*.zip', recursive=True)

        for model_zip_path in model_zip_paths:
            print("extracting zip", model_zip_path)
            output_dir = os.path.join('/content/so-vits-svc/models', os.path.basename(os.path.splitext(model_zip_path)[0]).replace(" ", "_"))

            # clean and create output dir
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.mkdir(output_dir)
            input_base = os.path.dirname(model_zip_path)

            # clean input dir (if user stopped an earlier extract and we have dirty files)
            ckpts_pre = glob.glob(os.path.join(input_base, '**/*.pth'), recursive=True)
            jsons_pre = glob.glob(os.path.join(input_base, '**/config.json'), recursive=True)
            for cpkt in ckpts_pre:
                os.remove(cpkt)
            for json in jsons_pre:
                os.remove(json)

            # do the extract
            self.extract(model_zip_path)
            ckpts = glob.glob(os.path.join(input_base, '**/*.pth'), recursive=True)
            jsons = glob.glob(os.path.join(input_base, '**/config.json'), recursive=True)
            for ckpt in ckpts:
                shutil.move(ckpt, os.path.join(output_dir, os.path.basename(ckpt)))
            for json in jsons:
                shutil.move(json, os.path.join(output_dir, os.path.basename(json)))


if __name__ == "__main__":
    downloader = Downloader()
    print(downloader.hf_models.list_models())
    model_name = "Applejack (singing)"
    target_dir = "models/Applejack (singing)"
    downloader.download_model(model_name, target_dir)

    print("Finished!")
