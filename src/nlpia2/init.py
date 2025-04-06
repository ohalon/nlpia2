# init.py
# import json
import logging
from pathlib import Path
import shutil
from urllib.request import urlretrieve, urljoin

# import pandas as pd
# import numpy as np

from .constants import PKG_NAME, BIGDATA_DIR, HOME_DATA_DIR
# from .constants import NLTK_STOPWORDS_ENGLISH, STOPWORDS, STOPWORDS_DICT
# from .constants import TLDS_POPULAR, tld_popular
# from .constants import TLDS, tld_iana

# from qary.etl.netutils import DownloadProgressBar  # noqa

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


DATA_URL_PREFIX = f'https://gitlab.com/tangibleai/{PKG_NAME}/-/raw/main/' \
                  f'src/{PKG_NAME}/data/'
DATA_URL_SUFFIX = '?inline=false'
DATA_FILENAMES = [
    # 'secret_message_convolved_line_plot.csv',
    'constants/nltk_stopwords_english.json',
    'constants/tlds-from-iana.csv',
    'constants/uri-schemes.xhtml.csv',
    'quotes.yml',
]

# download all this data too
BIG_DATA_FILENAME_URLs = {
    'secret_message_convolved_line_plot.csv': DATA_URL_PREFIX + 'secret_message_convolved_line_plot.csv' + DATA_URL_SUFFIX,
}


def maybe_download(
        url=None, filename=None, filepath=None,
        destination_dir=None, expected_bytes=None,
        force=False):
    """ Download a file only if it has not yet been cached locally in ~/.qary-data/ HOME_DATA_DIR
    """
    assert filepath is None or filename is None, f"Cannot specify both filepath='{filepath}' and filename='{filename}', they are synonymous."
    filename = filepath if filename is None else filename
    log.warning(f'filename: {filename}')
    src_filepath = BIGDATA_DIR / filename
    log.warning(f'src_filepath: {src_filepath}')
    home_filepath = HOME_DATA_DIR / filename
    log.warning(f'home_filepath: {home_filepath}')
    log.warning(f'src_filepath.is_file(): {src_filepath.is_file()}')
    if not home_filepath.parent.is_dir():
        home_filepath.parent.mkdir(exist_ok=True, parents=True)
    if filename and src_filepath.is_file():
        if home_filepath.is_file():
            return home_filepath

        log.warning(f'Need to copy {src_filepath} to {home_filepath}')
        assert src_filepath.is_file()
        shutil.copy(
            src=src_filepath,
            dst=home_filepath,
            follow_symlinks=True)
        assert home_filepath.is_file()
        return home_filepath

    if url is None:
        try:
            url = urljoin(str(DATA_URL_PREFIX), str(filename))
        except ValueError:
            log.warning('maybe_download() positional arguments deprecated. please specify url or filename (relative file path)')
            filename, url = url, None
    if filename is None and url is not None:
        filename = url.split('/')[-1].split('?')[0].split(':')[-1]
    if destination_dir is None:
        destination_dir = Path(HOME_DATA_DIR)
    filepath = destination_dir / filename
    destination_dir, filename = filepath.parent, filepath.name

    if not destination_dir.exists():
        destination_dir.mkdir(parents=True, exist_ok=True)  # FIXME add , reporthook=DownloadProgressBar())

    local_data_filepath = Path(BIGDATA_DIR) / filename
    if local_data_filepath.is_file() and not filepath.is_file():
        # TODO: use shutil.copy() to avoid running out of memory on large files
        filepath.write_bytes(local_data_filepath.read_bytes())

    if force or not filepath.is_file():
        log.info(f"Downloading: {url} to {filepath}")
        filepath, _ = urlretrieve(str(url), str(filepath))
        log.info(f"Finished downloading '{filepath}'")

    statinfo = Path(filepath).stat()

    # FIXME: check size of existing files before downloading
    if expected_bytes is not None:
        if statinfo.st_size == expected_bytes:
            log.info(f"Found '{filename}' and verified expected {statinfo.st_size} bytes.")
        else:
            raise Exception(f"Failed to verify: '{filepath}'. Check the url: '{url}'.")
    else:
        log.info(f"Found '{filename}' ({statinfo.st_size} bytes)")

    return filepath


def download_important_data(filenames=DATA_FILENAMES, force=False):
    """ Iterate through the important data filenames and download them from gitlab to a local cache """
    for i, relpath in enumerate(filenames):
        relpath = Path(relpath)
        destination_dir = HOME_DATA_DIR / relpath.parent
        url = DATA_URL_PREFIX + str(relpath) + DATA_URL_SUFFIX
        log.debug(f'url={url}')
        log.debug(f'relpath={relpath}')
        log.debug(f'relpath.name={relpath.name}')
        maybe_download(url=url, filename=relpath.name, destination_dir=destination_dir)

# shutil.copytree(src=QARY_DATA_DIR, dst=conf.DATA_DIR, dirs_exist_ok=True)




#####################################################################################
# pugnlp.constants

# tld_iana = pd.read_csv(Path(DATA_DIR, 'constants', 'tlds-from-iana.csv'), encoding='utf8')
# tld_iana = dict(sorted(zip((tld.strip().lstrip('.') for tld in tld_iana.domain),
#                            [(sponsor.strip(), -1) for sponsor in tld_iana.sponsor]),
#                        key=lambda x: len(x[0]),
#                        reverse=True))


# uri_schemes_iana = sorted(pd.read_csv(Path(DATA_DIR, 'constants', 'uri-schemes.xhtml.csv'),
#                                       index_col=0).index.values,
#                           key=lambda x: len(str(x)), reverse=True)


def init(data_filenames=DATA_FILENAMES, force=False):

    download_important_data(data_filenames=data_filenames, force=force)
