import hashlib
import zipfile
from pathlib import Path
from typing import Optional
from urllib import request as request
import os
import os.path as osp
import errno

from tqdm import tqdm

GBFACTOR = float(1 << 30)


def get_file_etag_checksum(filename: int, chunk_size=8 * 1024 * 1024):
    md5s = []
    with open(filename, "rb") as f:
        for data in iter(lambda: f.read(chunk_size), b""):
            md5s.append(hashlib.md5(data).digest())
    m = hashlib.md5(b"".join(md5s))
    return f"{m.hexdigest()}-{len(md5s)}"


def get_file_md5(filepath: str, chunk_size=8 * 1024 * 1024):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def file_validation(downloaded_file, gt_hash) -> bool:
    if "-" in gt_hash and gt_hash == get_file_etag_checksum(downloaded_file):
        return True
    if "-" not in gt_hash and gt_hash == get_file_md5(downloaded_file):
        return True
    raise RuntimeError("Downloaded file hash doesn't match reference.")


def get_file_size(url: str) -> int:
    """Returns file size in bytes

    Args:
        url (str): url of file to download
    """
    req = request.Request(url, method="HEAD")
    with request.urlopen(req) as f:
        file_size = f.headers.get("Content-Length")
    return int(file_size)


def tqdm_download_hook(t):
    """Wraps TQDM progress bar instance"""
    last_block = [0]

    def update_to(blocks_count: int, block_size: int, total_size: int):
        """Adds progress bar for request.urlretrieve() method
        Args:
            - blocks_count (int): transferred blocks count.
            - block_size (int): size of block in bytes.
            - total_size (int): size of requested file.
        """
        if total_size in (None, -1):
            t.total = total_size
        displayed = t.update((blocks_count - last_block[0]) * block_size)
        last_block[0] = blocks_count
        return displayed

    return update_to


def download_file(url: str, savepath: Path, true_hash: Optional[str], desc: str = None):
    if not savepath.parent.exists():
        savepath.parent.mkdir(parents=True, exist_ok=True)
    file_size = get_file_size(url)
    with tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        total=file_size,
        desc=desc,
    ) as t:
        request.urlretrieve(url, savepath, reporthook=tqdm_download_hook(t))
    if true_hash:
        file_validation(savepath, true_hash)

def decide_download(url):
    d = request.urlopen(url)
    size = int(d.info()["Content-Length"])/GBFACTOR

    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N)\n" % (size)).lower() == "y"
    else:
        return True

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)
    data = request.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    except:
        if osp.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')


    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)

