# string_normalizers.py
import string
import unicodedata
import re
from unidecode import unidecode


ASCII_LETTERS = string.ascii_letters
ASCII_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
ASCII_PRINTABLE_COMMON = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r'

ASCII_VERTICAL_TAB = '\x0b'
ASCII_PAGE_BREAK = '\x0c'
ASCII_ALL = ''.join(chr(i) for i in range(0, 128))  # ASCII_PRINTABLE
ASCII_DIGITS = string.digits
ASCII_IMPORTANT_PUNCTUATION = " .?!,;'-=+)(:"
ASCII_NAME_PUNCTUATION = " .,;'-"
ASCII_NAME_CHARS = set(ASCII_LETTERS + ASCII_NAME_PUNCTUATION)
ASCII_IMPORTANT_CHARS = set(ASCII_LETTERS + ASCII_IMPORTANT_PUNCTUATION)

CURLY_SINGLE_QUOTES = '‘’`´'
STRAIGHT_SINGLE_QUOTES = "'" * len(CURLY_SINGLE_QUOTES)
CURLY_DOUBLE_QUOTES = '“”'
STRAIGHT_DOUBLE_QUOTES = '"' * len(CURLY_DOUBLE_QUOTES)

from nlpia2.string_encoding import ENCODINGS  # noqa


class Asciifier:
    """ Construct a function that filters out all non-ascii unicode characters

    >>> test_str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
    >>> Asciifier(include='a b c 123XYZ')(test_str):
    '123abcXYZ '
    """

    def __init__(
            self,
            min_ord=1, max_ord=128,
            exclude=None,
            include=ASCII_PRINTABLE,
            exclude_category='Mn',
            normalize_quotes=True,
    ):
        self.include = set(sorted(include or ASCII_PRINTABLE))
        self._include = ''.join(sorted(self.include))
        self.exclude = exclude or set()
        self.exclude = set(sorted(exclude or []))
        self._exclude = ''.join(self.exclude)
        self.min_ord, self.max_ord = int(min_ord), int(max_ord or 128)
        self.normalize_quotes = normalize_quotes

        if self.min_ord:
            self.include = set(c for c in self.include if ord(c) >= self.min_ord)
        if self.max_ord:
            self.include = set(c for c in self._include if ord(c) <= self.max_ord)
        if exclude_category:
            self.include = set(
                c for c in self._include if unicodedata.category(c) != exclude_category)

        self.vocab = sorted(self.include - self.exclude)
        self._vocab = ''.join(self.vocab)
        self.char2i = {c: i for (i, c) in enumerate(self._vocab)}

        self._translate_from = self._vocab
        self._translate_to = self._translate_from

        # FIXME: self.normalize_quotes is accomplished by unidecode.unidecode!!
        # ’->'  ‘->'  “->"  ”->"
        if self.normalize_quotes:
            trans_table = str.maketrans(
                CURLY_SINGLE_QUOTES + CURLY_DOUBLE_QUOTES,
                STRAIGHT_SINGLE_QUOTES + STRAIGHT_DOUBLE_QUOTES)
            self._translate_to = self._translate_to.translate(trans_table)
            # print(self._translate_to)

        # eliminate any non-translations (if from == to)
        self._translate_from_filtered = ''
        self._translate_to_filtered = ''

        for c1, c2 in zip(self._translate_from, self._translate_to):
            if c1 == c2:
                continue
            else:
                self._translate_from_filtered += c1
                self._translate_to_filtered += c2

        self._translate_del = ''
        for c in ASCII_ALL:
            if c not in self.vocab:
                self._translate_del += c

        self._translate_from = self._translate_from_filtered
        self._translate_to = self._translate_to_filtered
        self.translation_table = str.maketrans(
            self._translate_from,
            self._translate_to,
            self._translate_del)

    def __call__(self, text):
        return unidecode(unicodedata.normalize('NFD', text)).translate(self.translation_table)


class TaggedStr(str):
    decoded_with_ = None

    def __init__(self, *args, **kwargs):
        # FIXME: can't used named kwarg when initializing a string?
        self.decoded_with_ = kwargs.pop('decoded_with_', None)
        super().__init__()


asciify = Asciifier()


def normalize_newlines(s):
    s = s.replace(ASCII_VERTICAL_TAB, '\n')
    s = s.replace(ASCII_PAGE_BREAK, '\n\n')


fold_newlines = normalize_newlines


def normalize_varname(s):
    """ Create a valid ASCII variable name from str s 

    >>> normalize_varname("\tGreat Journalism!\n--Maria (Ressa)  ")
    'great_journalism_maria_ressa'
    """
    s = asciify(s).strip().lower()
    return re.subn(r'[^\w]+', ' ', s)[0].strip().replace(' ', '_')


fold_varname = normalize_varname


def normalize_df_colnames(df):
    """ Return the DataFrame with asciified, lowered, striped column names """
    df.columns = [normalize_varname(c) for c in df.columns]
    return df


fold_colnames = normalize_df_colnames  # noqa


class Decoder:
    r"""
    https://docs.python.org/3/library/codecs.html#standard-encodings

    `errors`:
      strict  Raise UnicodeError (or a subclass), this is th...
      ignore  Ignore the malformed data and continue without...
      replace  Replace with a replacement marker. On encoding...
      backslashreplace  Replace with backslashed escape sequences. On ...
      surrogateescape  On decoding, replace byte with individual surr...,
      xmlcharrefreplace  Replace with XML/HTML numeric character refere...
      namereplace  Replace with \N{...} escape sequences, what ap...,
    """

    def __init__(self,
                 encodings=['utf-8', 'latin_1', 'ascii']  # python default, windows default
                 + ['iso8859_{i}' for i in range(2, 17)]  # various latin_# flavors
                 + ['utf-16', 'utf-64'],
                 errors='backslashreplace'
                 ):
        self.encodings = ENCODINGS if encodings == 'all' else encodings
        self.errors = errors

    def __call__(self, byts):
        for enc in self.encodings:
            try:
                s = byts.decode(enc)
                s = TaggedStr(s)
                s.decoded_with_ = enc
                return s
            except UnicodeDecodeError:
                pass
        s = byts.decode(errors=self.errors)
        s.decoder_ = 'undefined (errors={self.errors})'
        return s


try_decode = Decoder()
