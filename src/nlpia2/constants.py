# -*- coding: utf-8 -*-
"""Constants and discovered values, like path to current installation of nlpia2."""
import re
from pathlib import Path
import logging
import pkg_resources
from os import linesep, environ
import string
import datetime
from pytz import timezone
from collections.abc import Mapping

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from decimal import Decimal

# TODO: move constants.py->constants/__init__.py and constants_*.py to constants/*.py
from .constants_stopwords import NLTK_STOPWORDS_ENGLISH, STOPWORDS, STOPWORDS_DICT  # noqa
from .constants_uri_schemes import URI_SCHEMES, uri_schemes_iana  # noqa

log = logging.getLogger(__name__)

load_dotenv()
ENV = dict(environ)

EOL = LINESEP = linesep
HTML_TAGS = '<HTML', '<A HREF=', '<P>', '<BOLD>', '<SCRIPT', '<DIV', '<TITLE', '<BODY', '<HEADER'

PKG_DIR = Path(__file__).absolute().resolve().parent
PKG_NAME = PKG_DIR.name
SRC_DIR = PKG_DIR.parent
BASE_DIR = SRC_DIR.parent

HOME_DIR = Path.home().resolve().absolute()

SRC_DATA_DIR_NAME = 'data'
# nlpia2/src/nlpia2/data
SRC_DATA_DIR = PKG_DIR / SRC_DATA_DIR_NAME
try:
    SRC_DATA_DIR.mkdir(exist_ok=True, parents=True)
except FileExistsError as e:
    log.warninge(e)

BIGDATA_DIR_NAME = '.nlpia2-data'
BIGDATA_DIR = HOME_DATA_DIR = HOME_DIR / BIGDATA_DIR_NAME
try:
    BIGDATA_DIR.mkdir(exist_ok=True, parents=True)
except FileExistsError as e:
    log.warning(e)

# nlpia2/src/nlpia2/data/manuscript/adoc
MANUSCRIPT_DIR = SRC_DATA_DIR / 'manuscript'
ADOC_DIR = MANUSCRIPT_DIR / 'adoc'
IMAGES_DIR = MANUSCRIPT_DIR / 'images'

# nlpia2/
OFFICIAL_MANUSCRIPT_DIR = BASE_DIR
OFFICIAL_RELATIVE_MANUSCRIPT_DIR = Path('nlpia-manuscript') / 'manuscript'
for i in range(6):
    OFFICIAL_MANUSCRIPT_DIR = OFFICIAL_MANUSCRIPT_DIR.parent
    log.info(f'OFFICIAL_MANUSCRIPT_DIR: {OFFICIAL_MANUSCRIPT_DIR}')
    if ((OFFICIAL_MANUSCRIPT_DIR / 'nlpia-manuscript' / '.git').is_dir()
            and (OFFICIAL_MANUSCRIPT_DIR / OFFICIAL_RELATIVE_MANUSCRIPT_DIR / 'adoc').is_dir()):
        break
    OFFICIAL_MANUSCRIPT_DIR = OFFICIAL_MANUSCRIPT_DIR / 'hobs'
    log.info(f'OFFICIAL_MANUSCRIPT_DIR: {OFFICIAL_MANUSCRIPT_DIR}')
    if ((OFFICIAL_MANUSCRIPT_DIR / 'nlpia-manuscript' / '.git').is_dir()
            and (OFFICIAL_MANUSCRIPT_DIR / OFFICIAL_RELATIVE_MANUSCRIPT_DIR / 'adoc').is_dir()):
        break
    OFFICIAL_MANUSCRIPT_DIR = OFFICIAL_MANUSCRIPT_DIR.parent
OFFICIAL_MANUSCRIPT_DIR = (OFFICIAL_MANUSCRIPT_DIR / OFFICIAL_RELATIVE_MANUSCRIPT_DIR).resolve()
log.info(f'OFFICIAL_MANUSCRIPT_DIR: {OFFICIAL_MANUSCRIPT_DIR}')
if not (OFFICIAL_MANUSCRIPT_DIR / 'adoc').is_dir():
    OFFICIAL_MANUSCRIPT_DIR = next(OFFICIAL_MANUSCRIPT_DIR.parent.parent.glob('**/manuscript/adoc')).parent
try:
    assert OFFICIAL_MANUSCRIPT_DIR.name == 'manuscript'
    assert OFFICIAL_MANUSCRIPT_DIR.parent.name == 'nlpia-manuscript', f"OFFICIAL_MANUSCRIPT_DIR.parent: {OFFICIAL_MANUSCRIPT_DIR.parent}"
    OFFICIAL_IMAGES_DIR = OFFICIAL_MANUSCRIPT_DIR / 'images'
    log.info(f'OFFICIAL_IMAGES_DIR: {OFFICIAL_IMAGES_DIR}')
    assert OFFICIAL_IMAGES_DIR.is_dir()
except AssertionError:
    OFFICIAL_MANUSCRIPT_DIR = SRC_DATA_DIR / 'manuscript'
    log.error(f'CHANGED OFFICIAL_MANUSCRIPT_DIR: {OFFICIAL_MANUSCRIPT_DIR}')
OFFICIAL_ADOC_DIR = OFFICIAL_MANUSCRIPT_DIR / 'adoc'

# TODO: shutil.copytree(from_path, to_path)


assert OFFICIAL_MANUSCRIPT_DIR.name == 'manuscript'
assert OFFICIAL_MANUSCRIPT_DIR.parent.name in 'nlpia-manuscript data'.split()
assert OFFICIAL_ADOC_DIR.is_dir()
assert OFFICIAL_ADOC_DIR.name == 'adoc'
assert OFFICIAL_ADOC_DIR.parent.name == 'manuscript'
print(OFFICIAL_ADOC_DIR)


log.debug(f'SRC_DATA_DIR: {SRC_DATA_DIR}\n'
          + f'MANUSCRIPT_DIR: {MANUSCRIPT_DIR}\n'
          + f'ADOC_DIR: {ADOC_DIR}'
          )

DATA_INFO_FILE = SRC_DATA_DIR / 'data_info.csv'
BIGDATA_INFO_FILE = SRC_DATA_DIR / 'bigdata_info.csv'
BIGDATA_INFO_LATEST = BIGDATA_INFO_FILE.with_suffix('.latest.csv')

CHECPOINT_DIR = CHECKPOINT_PATH = BIGDATA_DIR / 'checkpoints'
LOG_DIR = Path(BIGDATA_DIR) / 'log'
try:
    LOG_DIR.mkdir(exist_ok=True)
except (FileExistsError, FileNotFoundError) as e:
    log.warning(e)

CONSTANTS_DIR = Path(BIGDATA_DIR) / 'constants'
HISTORY_PATH = Path(BIGDATA_DIR) / 'history.yml'
try:
    CONSTANTS_DIR.mkdir(exist_ok=True)
except (FileExistsError, FileNotFoundError) as e:
    log.warning(e)

QUESTIONWORDS = set('who what when were why which how'.split()
                    + ['how come', 'why does', 'can i', 'can you', 'which way'])
QUESTION_STOPWORDS = QUESTIONWORDS | set(STOPWORDS)

SPECIAL_PUNC = {
    "—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '*', '−': '-',
    "’": "'", "‘": "'", "´": "'", "`": "'", '،': ',',
    '\u200b': ' ', '\xa0': ' ', '„': '', '…': ' ... ', '\ufeff': '',
}


def get_version():
    """ Look within setup.cfg for version = ... and within setup.py for __version__ = """
    version = '0.0.0'
    try:
        return pkg_resources.get_distribution(PKG_NAME)
    except Exception as e:
        log.error(e)
        log.warning(f"Unable to find {PKG_NAME} version so using {version}")
    return version

    # setup.cfg will not exist if package install in site-packages
    with (BASE_DIR / 'setup.cfg').open() as fin:
        for line in fin:
            matched = re.match(r'\s*version\s*=\s*([.0-9abrc])\b', line)
            if matched:
                return (matched.groups()[-1] or '').strip()


__version__ = get_version()


def no_tqdm(it, total=1, **kwargs):
    """ Do-nothing iterable wrapper to subsitute for tqdm when verbose==False """
    return it

# DONE: create nlpia2/init.py
# DONE: add maybe_download() to init.py
# TODO: all required data files up to chapter07
# TODO: add list of all required data files to init.py
# TODO: ensure all files are in HOME_DATA_DIR (DATA_DIR is just a subset)
# TODO: move DATA_DIR constant to data.py
# DATA_FILENAMES = dict(
#     DATA_DIR
# )


# TZ constants
DEFAULT_TZ = timezone('UTC')

MAX_LEN_FILEPATH = 1023  # on OSX `open(fn)` raises OSError('Filename too long') if len(fn)>=1024

ROUNDABLE_NUMERIC_TYPES = (float, int, Decimal, bool)
FLOATABLE_NUMERIC_TYPES = (float, int, Decimal, bool)
BASIC_NUMERIC_TYPES = (float, int)
NUMERIC_TYPES = (float, int, Decimal, complex, str)  # datetime.datetime, datetime.date
NUMBERS_AND_DATETIMES = (float, int, Decimal, complex, str)
SCALAR_TYPES = (float, int, Decimal, bool, complex, str)  # datetime.datetime, datetime.date
# numpy types are derived from these so no need to include numpy.float64, numpy.int64 etc
DICTABLE_TYPES = (Mapping, tuple, list)  # convertable to a dictionary (inherits Mapping or is a list of key/value pairs)
VECTOR_TYPES = (list, tuple)
PUNC = str(string.punctuation)

# synonyms for "count"
COUNT_NAMES = ['count', 'cnt', 'number', 'num', '#', 'frequency', 'probability', 'prob', 'occurences']
# 4 types of "histograms" and their canonical name/label
HIST_NAME = {
    'hist': 'hist', 'ff': 'hist', 'fd': 'hist', 'dff': 'hist', 'dfd': 'hist', 'gfd': 'hist', 'gff': 'hist', 'bfd': 'hist', 'bff': 'hist',  # noqa
    'pmf': 'pmf', 'pdf': 'pmf', 'pd': 'pmf',  # noqa  prob. mass/density function, prob. density
    'cmf': 'cmf', 'cdf': 'cmf',  # noqa
    'cfd': 'cfd', 'cff': 'cfd', 'cdf': 'cfd',  # noqa
}
HIST_CONFIG = {
    'hist': {
        'name': 'Histogram',  # frequency distribution, frequency function, discrete ff/fd, grouped ff/fd, binned ff/fd
        'kwargs': {'normalize': False, 'cumulative': False, },
        'index': 0,
        'xlabel': 'Bin',
        'ylabel': 'Count',
    },
    'pmf': {
        # PMFs have discrete, exact values as bins rather than ranges (finite bin widths)
        #   but this histogram configuration doesn't distinguish between PMFs and PDFs,
        #   since mathematically they have all the same properties.
        #    PDFs just have a range associated with each discrete value
        #    (which should be when integrating a PDF but not when summing a PMF where the "width" is uniformly 1)
        'name': 'Probability Mass Function',   # probability density function, probability distribution [function]
        'kwargs': {'normalize': True, 'cumulative': False, },
        'index': 1,
        'xlabel': 'Bin',
        'ylabel': 'Probability',
    },
    'cmf': {
        'name': 'Cumulative Probability',
        'kwargs': {'normalize': True, 'cumulative': True, },
        'index': 2,
        'xlabel': 'Bin',
        'ylabel': 'Cumulative Probability',
    },
    'cfd': {
        'name': 'Cumulative Frequency Distribution',
        'kwargs': {'normalize': False, 'cumulative': True, },
        'index': 3,
        'xlabel': 'Bin',
        'ylabel': 'Cumulative Count',
    },
}

# these may not all be the sames isinstance types, depending on the env
FLOAT_TYPES = tuple([t for t in set(np.sctypeDict.values()) if t.__name__.startswith('float')] + [float])
FLOAT_DTYPES = tuple(set(np.dtype(typ) for typ in FLOAT_TYPES))
INT_TYPES = tuple([t for t in set(np.sctypeDict.values()) if t.__name__.startswith('int')] + [int])
INT_DTYPES = tuple(set(np.dtype(typ) for typ in INT_TYPES))
NUMERIC_TYPES = tuple(set(list(FLOAT_TYPES) + list(INT_TYPES)))
NUMERIC_DTYPES = tuple(set(np.dtype(typ) for typ in NUMERIC_TYPES))

DATETIME_TYPES = tuple(
    [t for t in set(np.sctypeDict.values()) if t.__name__.startswith('datetime')]
    + [datetime.datetime, pd.DatetimeTZDtype, pd.Timestamp]
)
DATE_TYPES = (datetime.datetime, datetime.date)

# matrices can be column or row vectors if they have a single col/row
VECTOR_TYPES = (list, tuple, np.matrix, np.ndarray)
MAPPING_TYPES = (Mapping, pd.Series, pd.DataFrame)

# These are the valid dates for all 3 datetime types in python (and the underelying integer nanoseconds)
INT_MAX = INT64_MAX = 2 ** 63 - 1
INT_MIN = INT64_MIN = - 2 ** 63
UINT_MAX = UINT64_MAX = - 2 ** 64 - 1

INT32_MAX = 2 ** 31 - 1
INT32_MIN = - 2 ** 31
UINT32_MAX = - 2 ** 32 - 1

INT16_MAX = 2 ** 15 - 1
INT16_MIN = - 2 ** 15
UINT16_MAX = - 2 ** 16 - 1

# Pandas timestamps can handle nanoseconds? by python datetimestampes cannot.
MAX_TIMESTAMP = pd.Timestamp('2262-04-11 23:47:16.854775', tz='utc')
MIN_TIMESTAMP = pd.Timestamp(datetime.datetime(1677, 9, 22, 0, 12, 44), tz='utc')
ZERO_TIMESTAMP = pd.Timestamp('1970-01-01 00:00:00', tz='utc')

# to_pydatetime() rounds to microseconds, ignoring 807 nanoseconds available in other MAX TIMESTAMPs
MIN_DATETIME = MIN_TIMESTAMP.to_pydatetime()
MAX_DATETIME = MAX_TIMESTAMP.to_pydatetime()
MIN_DATETIME64 = MIN_TIMESTAMP.to_datetime64()
MAX_DATETIME64 = MAX_TIMESTAMP.to_datetime64()
INF = np.inf
NAN = np.nan
NAT = pd.NaT


# str constants
MAX_CHR = MAX_CHAR = chr(127)
APOSTROPHE_CHARS = "'`’"
UNPRINTABLE = ''.join(set(chr(i) for i in range(128)) - set(string.printable))
string.unprintable = UNPRINTABLE  # monkey patch so import string from this module if you want this!

NULL_VALUES = set(['0', 'None', 'null', "'"] + ['0.' + z for z in ['0' * i for i in range(10)]])
# if datetime's are 'repr'ed before being checked for null values sometime 1899-12-30 will come up
NULL_REPR_VALUES = set(['datetime.datetime(1899, 12, 30)'])
# to allow NULL checks to strip off hour/min/sec from string repr when checking for equality
MAX_NULL_REPR_LEN = max(len(s) for s in NULL_REPR_VALUES)

PERCENT_SYMBOLS = ('percent', 'pct', 'pcnt', 'pt', r'%')
FINANCIAL_WHITESPACE = ('Flat', 'flat', ' ', ',', '"', "'", '\t', '\n', '\r', '$')
FINANCIAL_MAPPING = (('k', '000'), ('M', '000000'))

# MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
# MONTH_PREFIXES = [m[:3] for m in MONTHS]
# MONTH_SUFFIXES = [m[3:] for m in MONTHS]
# SUFFIX_LETTERS = ''.join(set(''.join(MONTH_SUFFIXES)))


if __name__ == '__main__':
    assert MANUSCRIPT_DIR.is_dir()
    assert ADOC_DIR.is_dir()  # !/usr/bin/env python
