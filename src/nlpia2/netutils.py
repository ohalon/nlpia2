""" Transformer based chatbot dialog engine for answering questions """
import logging
import os
import boto3
import urllib
import dotenv
from itertools import product

from pathlib import Path
from boto3.session import Session
from boto3.s3.transfer import S3Transfer
from tqdm import tqdm
from botocore.exceptions import NoCredentialsError

from .constants import SRC_DATA_DIR as DATA_DIR
from .constants import BASE_DIR
from .text_processing.re_patterns import cre_url

HOME_DIR = Path.home()
# from .constants import ENV
ENV = {}
SPACES_ACCESS_KEY, SPACES_SECRET_KEY = ENV.get('SPACES_ACCESS_KEY'), ENV.get('SPACES_SECRET_KEY')
LARGE_FILES = {}


log = logging.getLogger('qary')


class DownloadProgressBar(tqdm):
    """ Utility class that adds tqdm progress bar to urllib.request.urlretrieve

    >>> filemeta = LARGE_FILES['floyd']
    >>> filename = Path(filemeta['path']).name
    >>> url  = filemeta['url']
    >>> dest_path  = str(Path(filemeta['path']))
    >>> with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as dpb:
    ...     urllib.request.urlretrieve(url, filename=dest_path, reporthook=dpb.update_to)
    ('...qary/data/corpora/wikipedia/floyd.pkl', <http.client.HTTPMessage...>)
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def looks_like_url(url):
    """ Simplified check to see if the text appears to be a URL.

    Similar to `urlparse` but much more basic.

    Returns:
      True if the url str appears to be valid.
      False otherwise.

    >>> url = looks_like_url("totalgood.org")
    >>> bool(url)
    True
    """
    if not isinstance(url, str):
        return False
    if not isinstance(url, str) or len(url) >= 1024 or not cre_url.match(url):
        return False
    return True


def download_if_necessary(
        url_or_name,
        dest_path=None):
    """ Loads the file found in the local DATA_DIR or download it from the LARGE_FILES url

    >>> download_if_necessary('')
    """
    if not url_or_name:
        return None
    file_meta = LARGE_FILES.get(url_or_name) or {'url': url_or_name}
    log.debug(f'file_meta: {file_meta}')
    url = file_meta['url']
    if not dest_path and not looks_like_url(url):
        if Path(url).is_file():
            dest_path = url
        if Path(DATA_DIR, url).is_file():
            dest_path = url
    dest_path = dest_path or file_meta.get('path')
    # TODO: walk down from URL filename to match up with directories in DATA_DIR to build dest path
    if not dest_path:
        dest_path = file_meta.get('path', Path(DATA_DIR, file_meta.get('filename', Path(url).name)))
    filename = Path(dest_path).name
    if Path(dest_path).is_file():
        with urllib.request.urlopen(url) as file:
            cloud_file_size = file.info()['Content-Length']
        local_file_size = str(dest_path.stat().st_size)
        if cloud_file_size == local_file_size:
            return dest_path
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as dpb:
        try:
            urllib.request.urlretrieve(url, filename=dest_path, reporthook=dpb.update_to)
        except (urllib.error.HTTPError, ValueError):
            log.error(f'Unable to download file from "{url}" to "{dest_path}" using file_meta: {file_meta}.')
            return None
    return dest_path


for basedir, subdir in product(
    (DATA_DIR, BASE_DIR, HOME_DIR),
        'digitalocean-nlpia.org digitalocean digitalocean-qary.ai'.split()):
    p = Path(basedir).joinpath(subdir)
    if p.joinpath('.env.prod').is_file():
        dotenv.load_dotenv(p)
        break


def connect(
        access_key_id=os.getenv('DO_ACCESS_KEY'),
        secret_access_key=os.getenv('DO_ACCESS_SECRET'),
        region_name='sfo2',
        spaces_name='tan',
        url=None):

    url = url or f'https://{spaces_name}.{region_name}.digitaloceanspaces.com'
    session = Session()
    client = session.client('s3',
                            region_name='sfo2',
                            endpoint_url=url,
                            aws_access_key_id=access_key_id,
                            aws_secret_access_key=secret_access_key)
    return client


def ls():
    # need to have setup boto.cfg
    s3 = boto3.resource('s3')
    b = s3.Bucket('some/path/')
    # return list(b.objects.all())
    return list(b.objects.filter(Prefix='some/path'))


def upload_file(
        source='scripts/docs/*.pkl',
        dest=None,
        access_key_id=os.getenv('DO_ACCESS_KEY'),
        secret_access_key=os.getenv('DO_ACCESS_SECRET'),
        region_name='sfo2',
        spaces_name='tan',
        url=None):

    client = connect(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region_name=region_name,
        spaces_name=spaces_name,
        url=url)

    dest = dest or source
    retval = client.upload_file(source, spaces_name, dest)
    return retval


def upload_to_digitalocean(file_to_upload, name_of_space, digital_ocean_path_and_newfolders, new_name_file_to_upload):
    """ Upload files to digital ocean

    See Also:
      * [DO Spaces Docs]
        (https://developers.digitalocean.com/documentation/spaces/)
      * [Brian Bilo]
        (https://medium.com/@brianobilo/uploading-files-to-digital-ocean-spaces-using-api-keys-and-boto3-1f24242f6cd7)
constants
    >>> FILENAME = 'unit_test_data.json'
    >>> FILEPATH = Path(DATA_DIR, 'testsets', FILENAME)
    >>> file_to_upload = str(FILEPATH)
    >>> new_name_file_to_upload = FILENAME
    >>> name_of_space = "midata"
    >>> digital_ocean_path_and_newfolders = 'public/testsets'
    >>> response = upload_to_digitalocean(
    ...     file_to_upload, name_of_space, digital_ocean_path_and_newfolders, new_name_file_to_upload)
    >>> ((response is None and not SPACES_ACCESS_KEY)
    ...  or response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    ... ) == 200
    True
    """

    # Initialize a session using DigitalOcean Spaces.
    session = Session()
    if not SPACES_ACCESS_KEY or not SPACES_SECRET_KEY:
        log.error(f"SPACES_ACCESS_KEY: {SPACES_ACCESS_KEY} or SPACES_SECRET_KEY: {SPACES_SECRET_KEY} not set.")
        return
    try:
        client = session.client('s3',
                                region_name='SF02',
                                endpoint_url='https://tan.sfo2.digitaloceanspaces.com/',
                                aws_access_key_id=SPACES_ACCESS_KEY,
                                aws_secret_access_key=SPACES_SECRET_KEY)
        transfer = S3Transfer(client)

        # follows digital_ocean_path_and_newfolders
        # this will create folders if they don't yet exist
        # the file's final name is new_name_file_to_upload
        transfer.upload_file(file_to_upload, name_of_space,
                             digital_ocean_path_and_newfolders + "/" + new_name_file_to_upload)

        # This makes the file you have uploaded public by default. comment out responce if wanting to upload privately
        response = client.put_object_acl(ACL='public-read', Bucket=name_of_space, Key="%s/%s" %
                                         (digital_ocean_path_and_newfolders, new_name_file_to_upload))
    except NoCredentialsError:
        return
    return response
