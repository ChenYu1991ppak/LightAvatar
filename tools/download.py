""" 自动下载底模

    底模管理文档：
"""
import os
import requests
from tqdm import tqdm

from tools.common import load_model_paths
from tools.common import DEFAULT_CONFIG_FILE


def auto_download_weights(cfg_path=DEFAULT_CONFIG_FILE):
    download_urls, save_paths = load_model_paths(cfg_path)
    _download(download_urls, save_paths)


def _download(urls: dict, paths: dict):
    if isinstance(paths, str):
        if not os.path.exists(paths):
            download_with_progress(urls, local_path=paths)
    elif isinstance(paths, dict):
        for k, v in paths.items():
            _download(urls[k], paths[k])


def download_with_progress(url, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', desc=f"Downloading {os.path.basename(url)}", unit_scale=True)
        with open(local_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")
