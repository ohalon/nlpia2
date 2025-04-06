# string_normalizers.py
import string
import unicodedata


ASCII_LETTERS = string.ascii_letters
ASCII_PRINTABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
ASCII_PRINTABLE_COMMON = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r'

ASCII_VERTICAL_TAB = '\x0b'
ASCII_PAGE_BREAK = '\x0c'
ASCII_ALL = ASCII_PRINTABLE
ASCII_DIGITS = string.digits
ASCII_IMPORTANT_PUNCTUATION = " .,;'-?!"
ASCII_NAME_PUNCTUATION = " .,;'-?!"
ASCII_NAME_LETTERS = set(ASCII_LETTERS + ASCII_NAME_PUNCTUATION)
ASCII_IMPORTANT_CHARS = set(ASCII_LETTERS + ASCII_IMPORTANT_PUNCTUATION)


ASCII_IMPORTANT_PUNCTUATION


def normalize_newlines(s):
    s = s.replace(ASCII_VERTICAL_TAB, '\n')
    s = s.replace(ASCII_PAGE_BREAK, '\n\n')


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
            include_category='Mn'
    ):
        self._include = self.include = set(list(include))
        self.exclude = set(list(exclude))
        self.min_ord, self.max_ord = int(min_ord), int(max_ord or 128)
        if self._include is None:
            self._include = ASCII_IMPORTANT_CHARS
        if self.exclude is not None:
            self._include = self._include - self.exclude
        if self.min_ord:
            self._include = set(c for c in self._include if ord(c) >= self.min_ord)
        if self.max_ord:
            self._include = set(c for c in self._include if ord(c) <= self.max_ord)
        if include_category:
            self._include = set(
                c for c in self._include if unicodedata.category(c) != include_category)
        self._include = set(list(self._include))

    def __call__(self, text):
        return ''.join(c for c in text if c in self._include)
