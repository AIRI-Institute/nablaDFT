from typing import Optional
import hashlib
from urllib import request as request

from tqdm import tqdm


def get_file_etag_checksum(filename: int, chunk_size=8 * 1024 * 1024):
    md5s = []
    with open(filename, 'rb') as f:
        for data in iter(lambda: f.read(chunk_size), b''):
            md5s.append(hashlib.md5(data).digest())
    m = hashlib.md5(b"".join(md5s))
    return '{}-{}'.format(m.hexdigest(), len(md5s))


def get_file_md5(filepath: str, chunk_size=8 * 1024 * 1024):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def file_validation(downloaded_file, gt_hash) -> bool:
    if '-' in gt_hash and gt_hash == get_file_etag_checksum(downloaded_file):
        return True
    if '-' not in gt_hash and gt_hash == get_file_md5(downloaded_file):
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
    """wraps TQDM progress bar instance"""
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


def download_file(url: str, savepath: str, true_hash: Optional[str], desc: str = None):
    file_size = get_file_size(url)
    with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            total=file_size,
            desc=desc,
    ) as t:
        request.urlretrieve(
            url, savepath, reporthook=tqdm_download_hook(t)
        )
    if true_hash:
        file_validation(savepath, true_hash)
