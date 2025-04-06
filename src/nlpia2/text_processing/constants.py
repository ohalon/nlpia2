import numpy as np
import pandas as pd
import datetime
from collections.abc import Mapping
import string


uri_schemes_popular = [
    'http', 'https', 'telnet', 'mailto',
    'udp', 'ftp', 'ssh', 'git', 'apt', 'svn', 'cvs', 'hg',
    'smtp', 'feed',
    'example', 'content',
    'gtalk', 'chrome-extension',
    'bitcoin'
]
uri_schemes_web = ['http', 'https']

# these may not all be the sames isinstance types, depending on the env
FLOAT_TYPES = tuple([t for t in set(np.sctypeDict.values()) if t.__name__.startswith('float')] + [float])
FLOAT_DTYPES = tuple(set(np.dtype(typ) for typ in FLOAT_TYPES))
INT_TYPES = tuple([t for t in set(np.sctypeDict.values()) if t.__name__.startswith('int')] + [int])
INT_DTYPES = tuple(set(np.dtype(typ) for typ in INT_TYPES))
NUMERIC_TYPES = tuple(set(list(FLOAT_TYPES) + list(INT_TYPES)))
NUMERIC_DTYPES = tuple(set(np.dtype(typ) for typ in NUMERIC_TYPES))

DATETIME_TYPES = [t for t in set(np.sctypeDict.values()) if t.__name__.startswith('datetime')]
DATETIME_TYPES.extend([datetime.datetime, pd.Timestamp])
DATETIME_TYPES = tuple(DATETIME_TYPES)

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

# Pandas timestamps can handle nanoseconds? but python datetimestamps cannot.
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
RE_PRINTABLE_EXTENDED_ASCII = '[' + ''.join([chr(i) for i in range(161, 256)]) + ']'
RE_PRINTABLE_ASCII = r'[0-9a-zA-Z!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~ \t\n\r\x0b\x0c]'
RE_NOT_PRINTABLE_ASCII = r'[^' + RE_PRINTABLE_ASCII[1:]

# From the WikiText2 corpus (train.txt + valid.txt + test.txt)
# WikiText2 contains the nonprintable Unicode character '\ufeff'
# ''.join(set(re.findall(RE_NOT_PRINTABLE_ASCII, wikitext2)))
COMMON_PRINTABLE_NONASCII_CHARS = (
    '機უć–プấ動ძṣệîỳëưÅ⅓≤隊ųĀსСه्ルŻง‑ŌūøμณÍ⅔ス〉аảხễの‘→яṅ…大حầαíńêž♯Öზ☉ュ†าµè½³ÞÚჯრтôÉ“ò\ufeff'
    'àáვรс่ยłḥвị′ú²გØصッĐăκơწắ̃ãìკ♭şʻŁçėÆล§ü±—اеớó″იđ戦”空ñิアčś攻ʿ～์к჻оัửァ¡Ü−リ€ī〈ოōšâل・γ₤ม殻ûÁṭ'
    '₹火°£キトც„ヴن×å礮’ā場⁄დö¥ตṯÎṃäก·éβ'
)

# ''.join(chr(i) for i in range(128) if chr(i) not in string.printable)
NOT_PRINTABLE_ASCII_CHARS = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0e\x0f' \
    '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f'
# FIXME: standardize all nlpia2 code to one var name
ASCII_UNPRINTABLE_CHRS = UNPRINTABLE_ASCII_CHARS = NOT_PRINTABLE_ASCII_CHARS
# Monkey patch:
string.unprintable = UNPRINTABLE_ASCII_CHARS  # noqa

DASH_CHARS = '-‑—−–'
RE_DASH_CHAR = '[' + DASH_CHARS + ']'

APOSTROPHE_CHARS = "'’‘ʻ`′"
RE_APOSTROPHE_CHAR = '[' + APOSTROPHE_CHARS + ']'

DOUBLEQUOTE_CHARS = '"“”″'
RE_DOUBLEQUOTE_CHAR = '[' + DOUBLEQUOTE_CHARS + ']'

QUOTE_CHARS = APOSTROPHE_CHARS + DOUBLEQUOTE_CHARS
RE_QUOTE_CHAR = '[' + QUOTE_CHARS + ']'

NULL_VALUES = set(['0', 'None', 'null', "'", ""] + ['0.' + z for z in ['0' * i for i in range(10)]])
# if datetime's are 'repr'ed before being checked for null values sometime 1899-12-30 will come up
NULL_REPR_VALUES = set(['datetime.datetime(1899, 12, 30)'])
# to allow NULL checks to strip off hour/min/sec from string repr when checking for equality
MAX_NULL_REPR_LEN = max(len(s) for s in NULL_REPR_VALUES)

PERCENT_SYMBOLS = ('percent', 'pct', 'pcnt', 'pt', r'%')
FINANCIAL_WHITESPACE = ('Flat', 'flat', ' ', ',', '"', "'", '\t', '\n', '\r', '$')
FINANCIAL_MAPPING = (('k', '000'), ('M', '000000'))
