#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" File utilities comparable to similarly named bash utils: rm_rf(), rm_f(), and mkdir_p()

dataset1.0 is in files like: PPE1.rar PPE2.zip PPE3.zip PP4.7zip
dataset2.0 is in gs:/Buckets/safety_monitoring/data/obj/supplemental/"""
import gzip
import io
import json
import logging
import re
from pathlib import Path
from os.path import expandvars
from tqdm import tqdm

from html2text import html2text
import numpy as np
import pandas as pd

from .constants import MAX_LEN_FILEPATH
from .constants import BASE_DIR, SRC_DATA_DIR, BIGDATA_DIR
from .constants import HTML_TAGS, EOL
from .constants import no_tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

###################################################################
# deprecated __future__ and past imports and os.path.* upgrades to Pathlib

def listdir(path, **kwargs):
    return list(Path(path).glob('*', **kwargs))

def isdir(path):
    return Path(path).is_dir()

def isfile(path):
    return Path(path).is_file()

def basename(path):
    return Path(path).name

def dirname(path):
    return Path(path).parent

def remove(path, **kwargs):
    return Path(path).unlink(**kwargs)

def expanduser(path, **kwargs):
    return Path(path).expanduser(**kwargs)

def abspath(path, **kwargs):
    return Path(path).absolute(**kwargs)

def exists(path, **kwargs):
    return Path(path).exists(**kwargs)

def resolve(path, **kwargs):
    return Path(path).resolve(**kwargs)

# deprecated __future__ and past imports and os.path.* upgrades to Pathlib
#####################################################################


def wc(f, verbose=False, nrows=None):
    r""" Count lines in a text file

    References:
        https://stackoverflow.com/q/845058/623735

    >>> with open(Path(SRC_DATA_DIR, 'dictionary_fda_drug_names.txt')) as fin:
    ...     print(wc(fin) == wc(fin) == 7037 == wc(fin.name))
    True
    >>> wc(fin.name)
    7037
    """
    tqdm_prog = tqdm if verbose else no_tqdm
    with ensure_open(f, mode='r') as fin:
        for i, line in tqdm_prog(enumerate(fin)):
            if nrows is not None and i >= nrows - 1:
                break
        # fin.seek(0)
        return i + 1


def ensure_str(s, fallback_fun=repr):
    r""" Ensure that s is a str and not a bytes (.decode() if necessary)

    >>> ensure_str(b"I'm 2. When I grow up I want to be a str!")
    "I'm 2. When I grow up I want to be a str!"
    >>> ensure_str(42)
    '42'
    """
    try:
        return s.decode()
    except AttributeError:
        if isinstance(s, str):
            return s
    if fallback_fun and callable(fallback_fun):
        return fallback_fun(s)  # create a python repr (str) of a non-bytes nonstr object
    return s


def ls(path, force=False):
    """ bash `ls -a`: List both file paths or directory contents (files and directories)

    >>> ls('.')
    [...]
    >>> ls('~/')
    [...]

    >>> __file__.endswith(Path('nlpia', 'futil.py'))
    True
    >>> ls(__file__).endswith(Path('nlpia', 'futil.py'))
    True
    """
    path = expand_filepath(path)
    log.debug('path={}'.format(path))
    if isfile(path):
        return path
    elif isdir(path):
        return listdir(path)
    elif not force:
        return listdir(path)
    try:
        return listdir(path)
    except IOError:
        pass


def ls_a(path, force=False):
    """ bash `ls -a`: List both file paths or directory contents (files and directories)

    >>> path = ls(__file__)
    >>> path.endswith(Path('nlpia', 'futil.py'))
    True
    """
    return ls(path, force=force)


def find_data_path(path):
    for fullpath in [path,
                     Path(SRC_DATA_DIR, path),
                     Path(BIGDATA_DIR, path),
                     Path(BASE_DIR, path),
                     expanduser(Path('~', path)),
                     abspath(Path('.', path))
                     ]:
        if exists(fullpath):
            return fullpath
    return None


def expand_filepath(filepath):
    """ Expand any '~', '.', '*' variables in filepath.

    See also: pugnlp.futil.expand_path

    >>> len(expand_filepath('~')) > 3
    True
    """
    return resolve(abspath(expandvars(expanduser(filepath))))


def ensure_open(f, mode='r'):
    r""" Return a file pointer using gzip.open if filename ends with .gz otherwise open()

    TODO: try to read a gzip rather than relying on gz extension, likewise for zip and other formats
    TODO: monkey patch the file so that .write_bytes=.write and .write writes both str and bytes

    >>> fn = Path(SRC_DATA_DIR, 'pointcloud.csv.gz')
    >>> fp = ensure_open(fn)
    >>> fp
    <gzip _io.BufferedReader name='...src/nlpia/data/pointcloud.csv.gz' 0x...>
    >>> fp.closed
    False
    >>> with fp:
    ...     print(len(fp.readlines()))
    48485
    >>> fp.read()
    Traceback (most recent call last):
      ...
    ValueError: I/O operation on closed file
    >>> len(ensure_open(fp).readlines())
    48485
    >>> fn = Path(SRC_DATA_DIR, 'mavis-batey-greetings.txt')
    >>> fp = ensure_open(fn)
    >>> len(fp.read())
    314
    >>> len(fp.read())
    0
    >>> len(ensure_open(fp).read())
    0
    >>> fp.close()
    >>> len(fp.read())
    Traceback (most recent call last):
      ...
    ValueError: I/O operation on closed file.
    """
    fin = f
    if isinstance(f, (str, bytes)):
        f = ensure_str(f)
        if len(f) <= MAX_LEN_FILEPATH:
            f = find_filepath(f) or f
            if f and (not hasattr(f, 'seek') or not hasattr(f, 'readlines')):
                if f.lower().endswith('.gz'):
                    return gzip.open(f, mode=mode)
                return open(f, mode=mode)
            f = fin  # reset path in case it is the text that needs to be opened with StringIO
        else:
            f = io.StringIO(f)
    elif f and getattr(f, 'closed', None):
        if hasattr(f, '_write_gzip_header'):
            return gzip.open(f.name, mode=mode)
        else:
            return open(f.name, mode=mode)
    return f


def normalize_ext(filepath):
    """ Convert file extension(s) to normalized form, e.g. '.tgz' -> '.tar.gz'

    Normalized extensions are ordered in reverse order of how they should be processed.
    Also extensions are ordered in order of decreasing specificity/detail.
    e.g. zip last, then txt/bin, then model type, then model dimensionality

    .TGZ => .tar.gz
    .ZIP => .zip
    .tgz => .tar.gz
    .bin.gz => .w2v.bin.gz
    .6B.zip => .6B.glove.txt.zip
    .27B.zip => .27B.glove.txt.zip
    .42B.300d.zip => .42B.300d.glove.txt.zip
    .840B.300d.zip => .840B.300d.glove.txt.zip

    FIXME: Don't do this! Stick with the original file names and let the text loader figure out what it is!
    TODO: use regexes to be more general (deal with .300D and .42B extensions)

    >>> normalize_ext('glove.42B.300d.zip')
    'glove.42B.300d.glove.txt.zip'
    """
    mapping = tuple(reversed((
        ('.tgz', '.tar.gz'),
        ('.bin.gz', '.w2v.bin.gz'),
        ('.6B.zip', '.6b.glove.txt.zip'),
        ('.42B.zip', '.42b.glove.txt.zip'),
        ('.27B.zip', '.27b.glove.txt.zip'),
        ('.300d.zip', '.300d.glove.txt.zip'),
    )))
    if not isinstance(filepath, str):
        return [normalize_ext(fp) for fp in filepath]
    if '~' == filepath[0] or '$' in filepath:
        filepath = expand_filepath(filepath)
    fplower = filepath.lower()
    for ext, newext in mapping:
        r = ext.lower().replace('.', r'\.') + r'$'
        r = r'^[.]?([^.]*)\.([^.]{1,10})*' + r
        log.debug(f'regex pattern = {r}, string={filepath}')
        if re.match(r, fplower) and not fplower.endswith(newext):
            filepath = filepath[:-len(ext)] + newext
    return filepath


def normalize_filepath(filepath):
    r""" Lowercase the filename and ext, expanding extensions like .tgz to .tar.gz.

    >>> normalize_filepath('/Hello_World.txt\n')
    'hello_world.txt'
    >>> normalize_filepath('NLPIA/src/nlpia/bigdata/Goog New 300Dneg\f.bIn\n.GZ')
    'NLPIA/src/nlpia/bigdata/goog new 300dneg.bin.gz'
    """
    filename = basename(filepath)
    dirpath = filepath[:-len(filename)]
    cre_controlspace = re.compile(r'[\t\r\n\f]+')
    new_filename = cre_controlspace.sub('', filename)
    if not new_filename == filename:
        log.warning('Stripping whitespace from filename: {} => {}'.format(
            repr(filename), repr(new_filename)))
        filename = new_filename
    filename = filename.lower()
    filename = normalize_ext(filename)
    if dirpath:
        dirpath = dirpath[:-1]  # get rid of the trailing sep
        return Path(dirpath, filename)
    return filename


def find_filepath(
        filename,
        basepaths=(Path.cwd(), SRC_DATA_DIR, BIGDATA_DIR, BASE_DIR, '~', '~/Downloads', Path('/', 'tmp'), '..')):
    """ Given a filename or path see if it exists in any of the common places datafiles might be

    >>> p = find_filepath('iq_test.csv')
    >>> p == expand_filepath(Path(SRC_DATA_DIR, 'iq_test.csv'))
    True
    >>> p[-len('iq_test.csv'):]
    'iq_test.csv'
    >>> find_filepath('exponentially-crazy-filename-2.718281828459045.nonexistent')
    False
    """
    if isfile(filename):
        return filename
    for basedir in basepaths:
        fullpath = expand_filepath(Path(basedir, filename))
        if isfile(fullpath):
            return fullpath
    return False


def update_dict_types(d, update_keys=True, update_values=True, typ=(int,)):
    di = {}
    if not isinstance(typ, tuple):
        typ = (typ, )
    for k, v in d.items():
        ki, vi = k, v
        for t in typ:  # stop coercing type when the first conversion works
            if update_values and vi is v:
                try:
                    vi = t(v)
                except ValueError:
                    pass
            if update_keys and ki is k:
                try:
                    ki = t(k)
                except ValueError:
                    pass
        di[ki] = vi
    d.update(di)
    return d


def read_json(filepath, intkeys=True, intvalues=True):
    """ read text from filepath (`open(find_filepath(expand_filepath(fp)))`) then json.loads()

    >>> read_json('HTTP_1.1  Status Code Definitions.html.json')
    {'100': 'Continue',
     '101': 'Switching Protocols',...
    """
    d = json.load(ensure_open(find_filepath(filepath), mode='rt'))
    d = update_dict_types(d, update_keys=intkeys, update_values=intvalues)
    return d


def looks_like_index(series, index_names=('Unnamed: 0', 'pk', 'index', '')):
    """ Tries to infer if the Series (usually leftmost column) should be the index_col

    >>> looks_like_index(pd.Series(np.arange(100)))
    True
    """
    if series.name in index_names:
        return True
    if (series == series.index.values).all():
        return True
    if (series == np.arange(len(series))).all():
        return True
    if (
        (series.index == np.arange(len(series))).all() and
        str(series.dtype).startswith('int') and
        (series.count() == len(series))
    ):
        return True
    return False


def read_csv(*args, **kwargs):
    """Like pandas.read_csv, only little smarter: check left column to see if it should be the index_col

    >>> read_csv(Path(SRC_DATA_DIR, 'mavis-batey-greetings.csv')).head()
                                                    sentence  is_greeting
    0     It was a strange little outfit in the cottage.            0
    1  Organisation is not a word you would associate...            0
    2  When I arrived, he said: "Oh, hello, we're bre...            0
    3                                       That was it.            0
    4                I was never really told what to do.            0
    """
    kwargs.update({'low_memory': False})
    if isinstance(args[0], pd.DataFrame):
        df = args[0]
    else:
        log.info('Reading CSV with `read_csv(*{}, **{})`...'.format(args, kwargs))
        df = pd.read_csv(*args, **kwargs)
    if looks_like_index(df[df.columns[0]]):
        df = df.set_index(df.columns[0], drop=True)
        if df.index.name in ('Unnamed: 0', ''):
            df.index.name = None
    if ((str(df.index.values.dtype).startswith('int') and (df.index.values > 1e9 * 3600 * 24 * 366 * 10).any()) or
            (str(df.index.values.dtype) == 'object')):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            log.info('Unable to coerce DataFrame.index into a datetime using pd.to_datetime([{},...])'.format(
                df.index.values[0]))
    return df


def read_text(forfn, nrows=None, verbose=True):
    r""" Read all the lines (up to nrows) from a text file or txt.gz file

    >>> fn = Path(SRC_DATA_DIR, 'mavis-batey-greetings.txt')
    >>> len(read_text(fn, nrows=3))
    3
    """
    tqdm_prog = tqdm if verbose else no_tqdm
    nrows = wc(forfn, nrows=nrows)  # not necessary when nrows==None
    lines = np.empty(dtype=object, shape=nrows)
    with ensure_open(forfn) as f:
        for i, line in enumerate(tqdm_prog(f, total=nrows)):
            if i >= len(lines):
                break
            lines[i] = ensure_str(line).rstrip('\n').rstrip('\r')
        if all('\t' in line for line in lines):
            num_tabs = [sum([1 for c in line if c == '\t']) for line in lines]
            del lines
            if all(i == num_tabs[0] for i in num_tabs):
                f.seek(0)
                return read_csv(f, sep='\t', header=None, nrows=nrows)
        elif sum((1 for line in lines if any((tag.lower() in line.lower() for tag in HTML_TAGS)))
                 ) / float(len(lines)) > .05:
            return np.array(html2text(EOL.join(lines)).split(EOL))
    return lines
