from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os


def download_and_extract(url: str, path: str, model_dir: str) -> str:
    target_dir = os.path.join(path, model_dir)
    if not os.path.exists(target_dir):
        print(f'Model {model_dir} does not exist. Downloading {url} ...')
        resp = urlopen(url)
        with ZipFile(BytesIO(resp.read())) as zip_file:
            zip_file.extractall(path)

    assert os.path.exists(target_dir), f'{target_dir} was not present in downloaded file {url}'

    return target_dir
