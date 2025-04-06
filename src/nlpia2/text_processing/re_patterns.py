#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ^-- This allows unicode copypasta below
""" Compiled regular expressions for extracting dates, times, acronyms, etc

FIXME: Duplicate forms of regular expressions from master and develop branch need merging.

>>> CRE_ACRONYM.findall('National science Foundation (NSF)')
[('', '', '', '', 'National science Foundation', 'N', 's', 'F', 'NSF')]
>>> re.findall(RE_URL_SIMPLE, '* Sublime Text 3 (https://www.sublimetext.com/3) is great!')[0][0]
'https://www.sublimetext.com/3'
>>> re.findall(RE_URL_SIMPLE, 'Google github totalgood [github.com/totalgood]!')[0][0]
'github.com/totalgood'
"""
import copy
import logging
import os
import re
# import regex
import string
from pathlib import Path
import pandas as pd
from collections import OrderedDict
from nlpia2.text_processing.constants import APOSTROPHE_CHARS

log = logging.getLogger('qary')

try:
    DATA_DIR = Path(__file__).parent.parent / 'data'
except NameError:
    DATA_DIR = Path.cwd()
    print(f'USING CWD AS DATA_DIR: {DATA_DIR}')
#####################################################################################
#####################################################################################
# pugnlp.regexes

r"""Compiled regular expressions for tokenization and parsing
>>> list(m.group() for m in CRE_TOKEN.finditer("I'm sure \"Smiths'\" and \".net\" are easies; you?"))
["I'm", 'sure', '"', 'Smiths', '\'"', 'and', '"', '.', 'net', '"', 'are', 'easies', ';', 'you', '?']

RE_YEAR
>>> re.compile(RE_YEAR).findall("In '1970' and 2000, or 2015 '27 78 1886 with $1980 in my pocket")
['1970', '2000', '2015', '27', '78', '1980']
>>> doc = r"In '1970-2000\1', 2015/16, and 27, many not-so-wealthy people's banks had $1980 more than Gates' or Jobs'."
>>> ans = [next((s, LIST_RE_TOKEN_NAMED[i].lower()[3:]) for (i, s) in enumerate(groups) if s)
...        for groups in re.compile(RE_TOKEN_NAMED).findall(doc)]
>>> truth = [('In', 'unhyphenated_contracted_alpha'), ("'", 'nonword'), ('1970', 'year'),
... ('-', 'nonword'),('2000', 'year'), ('\\', 'nonword'),
... ('1', 'float'), ("',", 'nonword'), ('2015', 'year'), ('/', 'nonword'),
... ('16', 'year'), (',', 'nonword'), ('and', 'unhyphenated_contracted_alpha'),
... ('27', 'year'), (',', 'nonword'), ('many', 'unhyphenated_contracted_alpha'),
... ('not-so', 'hyphenated_alpha'), ('-', 'nonword'),
... ('wealthy', 'unhyphenated_contracted_alpha'), ("people's", 'unhyphenated_contracted_alpha'),
... ('banks', 'unhyphenated_contracted_alpha'),
... ('had', 'unhyphenated_contracted_alpha'), ('$1980', 'usd'), ('more', 'unhyphenated_contracted_alpha'),
... ('than', 'unhyphenated_contracted_alpha'),
... ('Gates', 'unhyphenated_contracted_alpha'), ("'", 'nonword'), ('or', 'unhyphenated_contracted_alpha'),
... ('Jobs', 'unhyphenated_contracted_alpha'),
... ("'.", 'nonword')]
>>> all([a == t for (a, t) in zip(ans, truth)])
True


RE_WORD_BASIC
  Disallows underscores,  hyphens, leading numerals, and leading punctuation (except dot e.g. ".Net").
  Trailing digits and mixed case accepted. Word break (\b) not required.
  >>> tough_words = "1on1 2_on_2\r19+2-1=4^2+2**2\tTitle9\n.Net SuperCalaFragalisticExpiAladozious Titles and a I"
  >>> ' '.join(m.group() for m in re.finditer(RE_WORD_BASIC, tough_words))
  'on1 on Title9 .Net SuperCalaFragalisticExpiAladozious Titles and a I'

RE_WORD_LIBERAL
  Allows underscores, hyphens, digits anywhere (trailing or leading).
  >>> ' '.join(iter_finds(RE_WORD_LIBERAL, tough_words))
  '1on1 2_on_2 19 2-1 4 2 2 2 Title9 .Net SuperCalaFragalisticExpiAladozious Titles and a I'

RE_WORD_ALGEBRA
  Underscores, hyphens, digits, and math operators allowed anywhere, but no whitespace
  >>> ' '.join(iter_finds(RE_WORD_ALGEBRA, tough_words))
  '1on1 2_on_2 19+2-1=4^2+2**2 Title9 .Net SuperCalaFragalisticExpiAladozious Titles and a I'

RE_WORD_UNDERSCORED
  2 to 3 "words" joined by internal underscores is an underscored word
  If external_underscores aren't matched by some preceding regex
  >>> compound_words = "Not-so-crazy words _underscored_externally_ and_internally and-very-long-up-to-64"
  >>> ' '.join(iter_finds(RE_WORD_UNDERSCORED, compound_words))
  'underscored_externally and_internally'

RE_PHRASE_UNDERSCORED
  4 to 64 "words" joined by internal underscores is a "PHRASE", like the title of a book or file
  >>> ' '.join(iter_finds(RE_PHRASE_UNDERSCORED, compound_words))
  ''

Only CAMEL_LIBERAL can start or end with an ACRONymn
But RE_ACRONYM only allows 5-char long acronyms, max. But the 6th can be the start of a title-case word.
RE_CAMEL_NORMAL
>>> [re.match(RE_CAMEL_NORMAL, s).group() for s in ['redRising', 'GoldenChildMorningStar']]
['redRising', 'GoldenChildMorningStar']
>>> list(re.finditer(RE_CAMEL, 'Morning5star Investing'))
[]
>>> list(m.group() for m in re.finditer(RE_CAMEL_NORMAL, 'EPR: AlbertEinstein BorisPodolsky And NathanRosen'))
['AlbertEinstein', 'BorisPodolsky', 'NathanRosen']

RE_CAMEL_LIBERAL, RE_CAMEL_LIBERAL_B
>>> list(m.group() for m in re.finditer(RE_CAMEL_LIBERAL, 'EinsteinPR: Einstein bPodolskyNRA NOTNRAPodolsky'))
['EinsteinPR', 'bPodolskyNRA', 'NOTNRAPodolsky']
>>> list(m.group() for m in re.finditer(RE_CAMEL_LIBERAL_B, 'EinsteinPR: Einstein bPodolskyNRA NOTNRAPodolsky'))
['EinsteinPR', 'bPodolskyNRA', 'NOTNRAPodolsky']
>>> [groups[0] for groups in re.findall(RE_CAMEL_LIBERAL_B, 'EinsteinPR: Einstein bPodolskyNRA NOTNRAPodolsky')]
['EinsteinPR', 'bPodolskyNRA', 'NOTNRAPodolsky']

FIXME: too narrow! probably because of all the \b checks
RE_DOTTED_ACRONYM_B
>>> list((m.group() if m else None) for m in re.finditer(RE_DOTTED_ACRONYM_B, 'U.S., U.S.A., A., and B.'))
['U.', 'U.S.']

RE_ACRONYM, RE_ACRONYM_B
>>> re.findall(RE_ACRONYM, 'Hello ACRNYM cANDid ATe')
['ACRNYM', 'AND', 'AT']
>>> re.findall(RE_ACRONYM_B, 'Hello ACRNYM cANDid ATe')
['ACRNYM']

RE_CAMEL_BASIC_B, RE_CAMEL_NORMAL_B, RE_CAMEL_LIBERAL_B
>>> [getattr(try_next(re.finditer(s, "Hello CamelACRONYM cANDid ATe")), 'group', bool)()
...  for s in (RE_CAMEL_BASIC_B, RE_CAMEL_NORMAL_B, RE_CAMEL_LIBERAL_B)]
[False, False, 'CamelACRONYM']
>>> scientific_notation_exponent.split(' 1 x 10 ** 23 ')
[' 1', '23 ']
>>> scientific_notation_exponent.split(' 1E10 and 1 x 10 ^23 ')
[' 1', '10 and 1', '23 ']
>>> scientific_notation_exponent.findall(' 1 x 10 ^23 ')
[' x 10 ^']
>>> scientific_notation_exponent.findall(' 1E10 and 1 x 10 ^23 ')
['E', ' x 10 ^']
>>> [bool(zero_pad_4_10_digit.match(an)) for an in
...  ['0000123744', '0', '0000', '0000000000', '0000001000', '000001', '0000126473', '000102952', '0000107079']]
[True, False, False, False, True, False, True, True, True]
>>> re_ver.match("__version__ = '0.0.18'").groups()
(None, '0', '0', '.18', '18', None, None)
>>> tweet = "Play the [postiive sum game](http://totalgood.com/a/b?c=42) of life instead of svn://us.gov."
>>> re.findall(url_popular, 'hello (hello.com/123/) whatever?')
[('hello.com/123/', '', '', 'hello.com', '.com', '/123/')]
>>> re.findall(url, 'hello (https://hello.com/123) whatever.?')
[('https://hello.com/123', 'https://', 'https', 'hello.com', 'com', '/123')]
>>> re.findall(url, 'hello (https://hello.com/123. whatever?')
[('https://hello.com/123.', 'https://', 'https', 'hello.com', 'com', '/123.')]
>>> re.findall(url, 'hello (https://hello.com/123/) whatever?')
[('https://hello.com/123/', 'https://', 'https', 'hello.com', 'com', '/123/')]
>>> re.findall(url, 'hello hello.com/123/. whatever?')
[('hello.com/123/.', '', '', 'hello.com', 'com', '/123/.')]
>>> re.findall(url, "What's this hello.com/123/? a url?")
[('hello.com/123/?', '', '', 'hello.com', 'com', '/123/?')]
>>> cre_url.findall(tweet)
[('http://totalgood.com/a/b?c=42', 'http://', 'http', 'totalgood.com', 'com', '/a/b?c=42'),
 ('svn://us.gov', 'svn://', 'svn', 'us.gov', 'gov', '')]
>>> cre_url_popular.findall(tweet)
[('http://totalgood.com/a/b?c=42', 'http://', 'http', 'totalgood.com', '.com', '/a/b?c=42'),
 ('svn://us.gov', 'svn://', 'svn', 'us.gov', 'gov', '')]
>>> list(match.groups()[0] for match in cre_url.finditer(tweet))
['http://totalgood.com/a/b?c=42', 'svn://us.gov']
>>> list(match.groups()[0] for match in re.finditer(url_popular, tweet))
['http://totalgood.com/a/b?c=42', 'svn://us.gov']

>>> tweet = "That site http://totalgood.com is awesome! don't you think? try.this.com? .Net ?"
>>> cre_url.findall(tweet)
[('http://totalgood.com', 'http://', 'http', 'totalgood.com', 'com', ''),
 ('try.this.com', '', '', 'try.this.com', 'com', '')]

>>> tweet = "Reach out to sombody.me (at) python.org if you like email@addresses.easy.com."
>>> list(match.groups()[0] for match in re.finditer(email_popular_obfuscated, tweet))
['sombody.me (at) python.org', 'email@addresses.easy.com']
>>> tweet = "What about dots in my {dot} name at python [dot] org?"
>>> list(match.groups()[0] for match in re.finditer(email_popular_obfuscated, tweet))
['my {dot} name at python [dot] org']
>>> re.match(at, r'  {at]  ').groups()[0]
'  {at]  '
>>> re.match(dot, ' \t(dot_').groups()[0]
' \t(dot_'
>>> re.match(at, r'@').groups()[0]
'@'
>>> re.match(dot, r'.').groups()[0]
'.'
>>> re.match(at, r'.')
>>> re.match(dot, r'@')
>>> re.match(username_obfuscated, 'hobson _.DOT._ lane hello world').groups()[0]
'hobson _.DOT._ lane'
"""

# from __future__ import division, print_function, absolute_import, unicode_literals
# from builtins import (bytes, dict, int, list, object, range, str,  # noqa
#     ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)

# try to make constants string variables all uppercase and regex patterns lowercase
ASCII_CHARACTERS = ''.join([chr(i) for i in range(128)])

list_bullet = re.compile(r'^\s*[! \t@#%.?(*+=-_]*[0-9.]*[#-_.)]*\s+')
nondigit = re.compile(r"[^0-9]")
nonphrase = re.compile(r"[^-\w\s/&']")
parenthetical_time = re.compile(r'([^(]*)\(\s*(\d+)\s*(?:min)?\s*\)([^(]*)', re.IGNORECASE)

tld_iana = pd.read_csv(DATA_DIR / 'tlds-from-iana.csv', encoding='utf8')
tld_iana = OrderedDict(sorted(zip((tld.strip().lstrip('.') for tld in tld_iana.domain),
                                  [(sponsor.strip(), -1) for sponsor in tld_iana.sponsor]),
                              key=lambda x: len(x[0]),
                              reverse=True))

tld_popular = OrderedDict(sorted([
    ('com', ('Commercial', 4860000000)),
    ('org', ('Noncommercial', 1950000000)),
    ('edu', ('US accredited postsecondary institutions', 1550000000)),
    ('gov', ('United States Government', 1060000000)),
    ('uk', ('United Kingdom', 473000000)),  # noqa
    ('net', ('Network services', 206000000)),
    ('ca', ('Canada', 165000000)),  # noqa
    ('de', ('Germany', 145000000)),  # noqa
    ('jp', ('Japan', 139000000)),  # noqa
    ('fr', ('France', 96700000)),  # noqa
    ('au', ('Australia', 91000000)),  # noqa
    ('us', ('United States', 68300000)),  # noqa
    ('ru', ('Russian Federation', 67900000)),  # noqa
    ('ch', ('Switzerland', 62100000)),  # noqa
    ('it', ('Italy', 55200000)),  # noqa
    ('nl', ('Netherlands', 45700000)),  # noqa
    ('se', ('Sweden', 39000000)),  # noqa
    ('no', ('Norway', 32300000)),  # noqa
    ('es', ('Spain', 31000000)),  # noqa
    ('mil', ('US Military', 28400000)),
    ], key=lambda x: len(x[0]), reverse=True))

break_path_lookahead = r'(?:\b|(?=[\s"\'>\].?!\)]))'
# break_path_lookahead = ''
fqdn_liberal = r'(\b[-.a-zA-Z0-9]+\b([.]' + r'|'.join(tld_iana) + ')\\b)'
# fqdn_liberal += break_path_lookahead
fqdn = fqdn_liberal
fqdn_popular = r'(\b[-.a-zA-Z0-9]+\b([.]' + r'|'.join(tld_popular) + ')\\b)'
# fqdn_popular += break_path_lookahead
username = r'(\b[-.a-zA-Z0-9!#$%&*+/=?^_`{|}~]+\b)'

email = re.compile(r'(\b' + username + r'\b@\b' + fqdn + '\\b)')
email_popular = re.compile(r'(\b' + username + r'\b@\b' + fqdn_popular + '\\b)')

# TODO: unmatched surrounding symbols are accepted/consumed, likewise for multiple dots/ats
at = r'(([-@="_(\[{\|\s]+(at|At|AT)[-@="_)\]\}\|\s]+)|[@])'
dot = r'(([-.="_(\[{\|\s]+(dot|dt|Dot|DOT)[-.="_)\]\}\|\s]+)|[.])'

tld_iana = r'(' + r'|'.join(tld_iana) + ')'
cre_tld_iana = re.compile(tld_iana)
tld_popular = r'(' + r'|'.join(tld_popular) + ')'
cre_tld_popular = re.compile(tld_popular)
fqdn_obfuscated = r'(\b(([-a-zA-Z0-9]+)' + dot + '{1,7})' + tld_iana + r'\b' + ')'
cre_fqdn_obfuscated = re.compile(fqdn_obfuscated)
fqdn_popular_obfuscated = r'(\b(([-a-zA-Z0-9]+' + dot + '){1,7})' + tld_popular + r'\b' + ')'
cre_fqdn_popular_obfuscated = re.compile(fqdn_popular_obfuscated)
username_obfuscated = r'((([a-zA-Z0-9!#$%&*+/?^`~]+)' + dot + '?){1,7})'
cre_username_obfuscated = re.compile(username_obfuscated)

cre_email_obfuscated = re.compile(r'(\b' + username_obfuscated + at + fqdn_obfuscated + r'\b' + ')')
cre_email_popular_obfuscated = re.compile(r'(\b' + username_obfuscated + at + fqdn_popular_obfuscated + r'\b)')

href = r'([Hh][Rr][Ee][Ff]\s?=\s?["\'])([^"\']+)'
cre_href = re.compile(href)

# doesn't allow for unescaped quoted or parenthesized query strings like:
#   ?x="1" ?x='1' ?x=(1) and ?x=[1]
uri_schemes_iana = sorted(pd.read_csv(DATA_DIR / 'uri-schemes.xhtml.csv',
                                      index_col=0).index.values,
                          key=lambda x: len(str(x)), reverse=True)
uri_schemes_popular = ['chrome-extension', 'example', 'content', 'bitcoin',
                       'telnet', 'mailto',
                       'https', 'gtalk',
                       'http', 'smtp', 'feed',
                       'udp', 'ftp', 'ssh', 'git', 'apt', 'svn', 'cvs']
url_path = r'(?:[/][^\s"\'\]\)]*' + break_path_lookahead + ')+'
url_path = r'(' + url_path + break_path_lookahead + ')'
url_scheme = r'(\b(' + '|'.join(uri_schemes_iana) + ')[:][/]{2})'
url_scheme_popular = r'(\b(' + '|'.join(uri_schemes_popular) + ')[:][/]{2})'

# allows paths to stop before trailing sentence period like: example.com/file. or example.com/file!
url_strict = r'(\b' + url_scheme + fqdn + url_path + '?)' + break_path_lookahead

url_popular_strict = r'(\b' + url_scheme + fqdn_popular + url_path + '?)' + break_path_lookahead
url_popular = r'(\b' + url_scheme + '?' + fqdn_popular + url_path + '?)' + break_path_lookahead
url_liberal = r'(\b' + url_scheme + '?' + fqdn_liberal + url_path + '?)' + break_path_lookahead

cre_url_strict = re.compile(url_strict)
cre_url_liberal = re.compile(url_liberal)

cre_url_popular_strict = re.compile(url_popular_strict)
cre_url_popular = re.compile(url_popular)

url = url_liberal
cre_url = re.compile(url)

nonword = re.compile(r'[\W]')
white_space = re.compile(r'[\s]')
# ASCII regexes from http://stackoverflow.com/a/20078869/623735
# To replace sequences of nonASCII characters with a single "?" use `nonascii_sequence.sub("?", s)`
nonascii_sequence = re.compile(r'[^\x00-\x7F]+')
# To replace sequences of nonASCII characters with a "?" per character use `nonascii.sub("?", s)`
nonascii = re.compile(r'[^\x00-\x7F]')
# To replace sequences of ASCII characters with a single "?" use `ascii_sequence.sub("?", s)`
ascii_sequence = re.compile(r'[^\x00-\x7F]+')
# To replace sequences of ASCII characters with a "?" per character use `ascii.sub("?", s)`
ascii_char_class = re.compile(r'[\x00-\x7F]')
# would be better-named as scientific_notation_base

scientific_notation_exponent = re.compile(r'\s*(?:[xX]{1}\s*10\s*[*^]{1,2}|[eE]){1}\s*')
nondigit = re.compile(r'[^\d]+')
not_digit_list = re.compile(r'[^\d,]+')
not_digit_nor_sign = re.compile(r'[^0-9-+]+')

word_sep_except_external_appostrophe = re.compile(r'\W*\s\'{1,3}|\'{1,3}\W+|[^-\'_.a-zA-Z0-9]+|\W+\s+')
word_sep_permissive = re.compile(r'[^\'a-zA-Z0-9]\s\W*|[^-\'_.a-zA-Z0-9]+')
sentence_sep = re.compile(r'[.?!](\W+)|$')
month_name = re.compile(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[acbeihmlosruty]*', re.IGNORECASE)


# A permissive filter of javascript variable/function names
#  Allows unicode and leading undercores and $
#  From http://stackoverflow.com/a/2008444/623735
js_name = re.compile(u'^[_$a-zA-Z\xA0-\uFFFF][_$a-zA-Z0-9\xA0-\uFFFF]*$')

# avoids special wikipedia URLs like ambiguity resolution pages
wikipedia_special = re.compile(r'.*wikipedia[.]org/wiki/[^:]+[:].*')

nones = re.compile(
    r'^Unk[n]?own|unk[n]?own|UNK|Unk|UNK[N]?OWN|[.]+|[-]+|[=]+|[_]+|[*]+|[?]+|N[/]A|n[/]a'
    r'|None|none|NONE|Null|null|NULL|NaN$')

# Unary NOT operator and its operand returned in match.groups() 2-tuple
not_symbol = re.compile(r'[Nn][Oo][Tt]|[\~\-\!\^]')
notter = re.compile(r'(' + not_symbol.pattern + r')?\s*(.*)\s*')

# A 4-10 digit numerical serial number or account number with zero padding
#   * Allow any number of padding zeros to precede the 4-10 "significant" digits
#   * Allow whitespace on both ends
#   * Allows '0000' but not '0001' or '0000000001'
zero_pad_4_10_digit = re.compile(r'[0]{0,6}[1-9][0-9]{3,9}')
serial_number = zero_pad_4_10_digit
account_number = zero_pad_4_10_digit

optionally_notted_zero_pad_4_10_digit = re.compile(
    r'\s*(' + not_symbol.pattern + ')?\\s*(' + zero_pad_4_10_digit.pattern + ')\\s*')

# python package version number specification (PEP 440: [N!]N(.N)*[{a|b|rc}N][.postN][.devN] )
re_ver = re.compile(r"^\s*[_]{0,2}version[_]{0,2}\s*=\s*\'(\d*!)?(\d+)\.(\d+)(\.(\d+))?((a|b|rc)\d*)?\'")

########################################################################
# for Twitter tweets

re_hashtag = r'([-\s!?.;]|^)(#[A-Za-z]{2,32})\b'
cre_hashtag = re.compile(re_hashtag)
re_atuser = r'([-\s!?.;]|^)(@[A-Za-z_0-9]{2,32})\b'
cre_atuser = re.compile(re_atuser)
re_hashtag_at_end = r'.*\s([#][A-Za-z]{2,32})\s*[.?!-=\s]{0,8}\s*$'
cre_hashtag_at_end = re.compile(re_hashtag_at_end)

# for Twitter tweets
########################################################################


#####################################################
# Sequence getters/iterators/wrappers


def iter_finds(regex_obj, s):
    """Generate all matches found within a string for a regex and yield each match as a string"""
    if isinstance(regex_obj, str):
        for m in re.finditer(regex_obj, s):
            yield m.group()
    else:
        for m in regex_obj.finditer(s):
            yield m.group()


def try_next(it, default=None):
    try:
        return next(it)
    except StopIteration:
        return default


def try_get(obj, idx, default=None):
    try:
        return obj.__getitem__(idx)
    except IndexError:
        return default


def wrap(s, prefix=r'\b', suffix=r'\b', grouper='()'):
    r"""Wrap a string (tyically a regex) with a prefix and suffix (usually a nonconuming word break)
    Arguments:
      prefix, suffix (str): strings to append to the front and back of the provided string
      grouper (2-len str or 2-tuple): characters or strings to separate prefix and suffix from the middle
    >>> wrap(r'\w*')
    '\\b(\\w*)\\b'
    >>> wrap(r'middle', prefix=None)
    '(middle)\\b'
    """
    wrapped = prefix or ''
    wrapped += try_get(grouper, 0, '')
    wrapped += s or ''
    wrapped += try_get(grouper, 1, try_get(grouper, 0, ''))
    return wrapped + (suffix or '')


# Sequence getters/iterators/wrapeers
######################################################

RE_BAD_FILENAME = '[{}]'.format(re.escape(string.punctuation + string.unprintable))
RE_PUNCT = '[{}]'.format(re.escape(string.punctuation))
RE_UPPER_CLASS = re.compile(r'[A-Z]')
RE_LOWER_CLASS = re.compile(r'[a-z]')
RE_DIGIT_CLASS = re.compile(r'[0-9]')
# \w = r'[a-zA-Z0-9_]'
RE_WORD_CLASS = r'[a-zA-Z0-9_]'

# numerals only allowed at the end of a word, but include it in the word
# hyphens and underscores only allowed at the end of letters before any numerals
# start with an optional dot, then have to have at least 1 letter
# opitonal numerals at the end of word segments, underscores and hyphens between word segments

RE_WORD = r'^([a-zA-Z][-_a-zA-Z]*[\w0-9])[\W]*$'
CRE_WORD = re.compile(RE_WORD)
# RE_WORD_UNGROUPED = r'[a-zA-Z][-_a-zA-Z]*[\w0-9]'

RE_WORD_BASIC = r'[.]?[a-zA-Z]+[0-9]*'
RE_WORD_BASIC_B = wrap(RE_WORD_BASIC)
RE_WORD_LIBERAL = r'[-.a-zA-Z0-9_]+'
RE_WORD_LIBERAL_B = wrap(RE_WORD_LIBERAL)
RE_WORD_CAPITALIZED = r'[A-Z][a-z]+[0-9]{0,3}'
RE_WORD_CAPITALIZED_B = r'\b(' + RE_WORD_CAPITALIZED + ')\\b'
RE_WORD_ACRONYM = r"[A-Z0-9][A-Z0-9]{1,6}[0-9]{0,2}"
RE_WORD_ACRONYM_B = r'\b(' + RE_WORD_ACRONYM + ')\\b'
RE_WORD_LOWERCASE = r'[a-z]+[0-9]{0,3}'
RE_WORD_LOWERCASE_B = r'\b(' + RE_WORD_LOWERCASE + ')\\b'
RE_CAMEL_BASIC = '(' + RE_WORD_CAPITALIZED + '){2,6}'
RE_CAMEL_BASIC_B = r'\b(' + RE_CAMEL_BASIC + ')\\b'
RE_CAMEL_BASIC_LONG = '(' + RE_WORD_CAPITALIZED + '){7,256}'
RE_CAMEL_BASIC_LONG_B = r'\b(' + RE_CAMEL_BASIC + ')\\b'
RE_CAMEL_NORMAL = '(' + RE_CAMEL_BASIC + ')|([a-z]+(' + RE_WORD_CAPITALIZED + '){1,5})'
RE_CAMEL_NORMAL_B = r'\b(' + RE_CAMEL_NORMAL + ')\\b'
RE_CAMEL_LIBERAL = (r'\b('
                    '(' + RE_CAMEL_NORMAL + ')|'
                    '(' + RE_WORD_ACRONYM + '(' + RE_WORD_CAPITALIZED + '){1,5}' + ')|'
                    '(' + '[a-z]{0,24}(' + RE_WORD_CAPITALIZED + '){1,5}' + RE_WORD_ACRONYM + ')'
                    ')\\b')
RE_CAMEL_LIBERAL_B = r'\b(' + RE_CAMEL_LIBERAL + ')\\b'
CRE_CAMEL_LIBERAL_B = re.compile(RE_CAMEL_LIBERAL_B)
RE_CAMEL = RE_CAMEL_LIBERAL
RE_CAMEL_B = RE_CAMEL_LIBERAL_B

CHARS_DASHES = "–"
CHARS_ALGEBRA = "-+*/^!=().a-zA-Z0-9_'"
RE_WORD_ALGEBRA = '[' + CHARS_ALGEBRA + ']+'

QUOTE_CHARS = "\"'`’“”"
RE_WORD_BASIC_QUOTED = '|'.join(c + RE_WORD_BASIC + c for c in QUOTE_CHARS)
RE_WORD_LIBERAL_QUOTED = '|'.join(c + RE_WORD_LIBERAL + c for c in QUOTE_CHARS)
RE_WORD_ALGEBRA_QUOTED = '|'.join(c + RE_WORD_ALGEBRA + c for c in QUOTE_CHARS)
RE_PHRASE_BASIC_QUOTED = '|'.join(c + '((' + RE_WORD_BASIC + ')|\\W)+' + r'\W?' + c for c in QUOTE_CHARS)
RE_PHRASE_LIBERAL_QUOTED = '|'.join(c + '((' + RE_WORD_LIBERAL + ')|\\W)+' + c for c in QUOTE_CHARS)
RE_PHRASE_ALGEBRA_QUOTED = '|'.join(c + '((' + RE_WORD_ALGEBRA + ')|\\W)+' + c for c in QUOTE_CHARS)

# 2 to 3 "words" joined by internal underscores is just an underscored word
RE_WORD_UNDERSCORED = '|'.join('[_]+'.join([RE_WORD_BASIC] * i) for i in range(2, 4))
# 4 to 64 "words" joined by internal underscores is a "PHRASE", like the title of a book or file
RE_PHRASE_UNDERSCORED = '|'.join('[_]+'.join([RE_WORD_BASIC] * i) for i in range(4, 65))
# 2 to 3 "words" joined by internal hyphens is just a hyphenated (compound) word
RE_WORD_HYPHENATED = '|'.join('[_]+'.join([RE_WORD_BASIC] * i) for i in range(2, 4))
# 4 to 64 "words" joined by internal hyphens is a "PHRASE"
RE_PHRASE_HYPHENATED = '|'.join('[_]+'.join([RE_WORD_BASIC] * i) for i in range(4, 65))

# based on pci/unused/chapter3/generatefeedvector.py
RE_HTML_TAG = r'[\s]*<[^>]+>[\s]*'
RE_DOUBLEQUOTE = r'["]+'
# \d = [0-9]  # also unicode numerals in all scripts (but only in unicode-supporting flavors unlike Java)
# \w = [a-zA-Z0-9_]

CHARS_LOWER = ''.join(chr(i) for i in range(ord('a'), ord('z') + 1))
CHARS_UPPER = ''.join(chr(i) for i in range(ord('A'), ord('Z') + 1))
CHARS_DIGIT = ''.join(chr(i) for i in range(ord('0'), ord('9') + 1))
CHARS_ALPHA = CHARS_LOWER + CHARS_UPPER
CHARS_ALPHANUM = CHARS_ALPHA + CHARS_DIGIT
RE_CLASS_ALPHANUM = '[a-zA-Z0-9]'

# Dots and allowed to delimit words, none of the 3 apostrophes nor & symbol do
RE_WORD_DELIM = r"[^-&a-zA-Z0-9_" + APOSTROPHE_CHARS + r"]"
# FIXME: Only single-hyphenated words are accecpted, unaccptable-multi-hyphenated words
RE_HYPHENATED_ALPHA = r"\w+\-\w+"
RE_HYPHENATED_ALPHA_B = r'\b(' + RE_HYPHENATED_ALPHA + ')\\b'
RE_HYPHENATED_ALPHANUM = r"[a-zA-Z]\w*\-\w*[a-zA-Z][0-9]*"
RE_HYPHENATED_ALPHANUM_B = r'\b(' + RE_HYPHENATED_ALPHANUM + ')\\b'
RE_DOT_PREFIXED_ALPHANUM = '[.]' + RE_WORD_BASIC
RE_DOT_PREFIXED_ALPHANUM_B = r'\b(' + RE_DOT_PREFIXED_ALPHANUM + ')\\b'
RE_DOT_PREFIXED_HYPHENATED_ALPHANUM = '[.]' + RE_HYPHENATED_ALPHANUM
RE_DOT_PREFIXED_HYPHENATED_ALPHANUM_B = r'\b(' + RE_DOT_PREFIXED_HYPHENATED_ALPHANUM + ')\\b'
# for .Net or .Netable
RE_HYPHENATED_DOTTED_ALPHANUM = r"[a-zA-Z]\w*[-.]\w*[a-zA-Z][0-9]*"
RE_HYPHENATED_DOTTED_ALPHANUM_B = r'\b(' + RE_HYPHENATED_DOTTED_ALPHANUM + ')\\b'

# FIXME: Plural words at end single quotes around plural words to be interpretted as possessive
RE_POSESSIVE_ALPHA = r"\w+'[sS]|\w+\-\w+[sS]'|\w+\-\w+"
RE_POSESSIVE_ALPHA_B = r'\b(' + RE_POSESSIVE_ALPHA + ')\\b'
RE_HYPHENATED_POSESSIVE_ALPHA = r"\w+\-\w+'[sS]|\w+\-\w+[sS]'|\w+\-\w+"
RE_HYPHENATED_POSESSIVE_ALPHA_B = r'\b(' + RE_HYPHENATED_POSESSIVE_ALPHA + ')\\b'

# This will accept a lot of mispelled or nonsense "contractions" and mis some odd, but valid ones listed here:
#    https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
RE_UNHYPHENATED_CONTRACTED_ALPHA = r"['`’]tis|['`’]twas|\w+['`’][a-zA-Z]{1,2}|\w+"
RE_UNHYPHENATED_CONTRACTED_ALPHA_B = r'\b(' + RE_UNHYPHENATED_CONTRACTED_ALPHA + ')\\b'
RE_USD_DECIMAL_BMK = r'\$\d+[.]\d+[BMKk]'
RE_USD_DECIMAL_BMK_B = r'\b(' + RE_USD_DECIMAL_BMK + ')\\b'
RE_USD_BMK = r'\$[\d]+[BMKk]'
RE_USD_BMK_B = r'\b(' + RE_USD_BMK + ')\\b'
RE_USD_CENTS = r'\$\d+[.]\d\d'  # don't allow decidollars or millidollars?
RE_USD_CENTS_B = r'\b(' + RE_USD_CENTS + ')\\b'
RE_USD = r'\$[\d]+'
RE_USD_B = r'\b(' + RE_USD + ')\\b'
# TODO: add EU and Asian Currencies and decimal formats (swap comma and decimal)
RE_FLOAT = r'[\d]+[.]?\d*'
RE_FLOAT_B = r'\b(' + RE_FLOAT + ')\\b'
RE_FLOAT_E = r'[\d]+[.]?\d*[ ]?[eE][ ]?\d+'
RE_FLOAT_E_B = r'\b(' + RE_FLOAT_E + ')\\b'
RE_NONSPACE = r'\S+'
RE_NONSPACE_B = r'\b(' + RE_NONSPACE + ')\\b'
RE_NONWORD = r'[^\s\w]+'
RE_NONWORD_B = r'\b(' + RE_NONWORD + ')\\b'
RE_YEAR = r"\b19\d\d\b|\b20\d\d\b|\b[']?\d\d\b"
RE_YEAR_B = r'\b(' + RE_YEAR + ')\\b'
RE_DECADE = r"\b19\d0[']?s\b|\b20\d0[']?s\b|\b[']?\d0[']?s\b"
RE_DECADE_B = r'\b(' + RE_DECADE + ')\\b'

RE_ACRONYM = r"[A-Z0-9][A-Z0-9]{1,5}[0-9]{0,2}"
RE_ACRONYM_B = r'\b(' + RE_ACRONYM + ')\\b'
# only very narrow, but common examples fit: U.S., U.S.A., A., and B.
RE_DOTTED_ACRONYM_B = r"\b[A-Z][.][A-Z][.][A-Z][.][A-Z][.]\b|\b[A-Z][.][A-Z][.][A-Z][.]\b|\b[A-Z][.][A-Z][.]\b|\b[A-Z][.]\b"
# RE_DOT_NET = r"\b[.]\w[.][A-Z][.][A-Z][.]\b|\b[A-Z][.][A-Z][.][A-Z][.]\b|\b[A-Z][.][A-Z][.]\b|\b[A-Z][.]\b"

# # Wrap token RE w/ parens (in case it contains ORs) and add nonconsuming word break (\b) at the end
# for name in ('WORD_CAPITALIZED', 'WORD_LOWERCASE', 'ACRONYM', 'USD_BMK', 'USD_CENTS', 'USD_DECIMAL_BMK',
#              'POSESSIVE_ALPHA', 'HYPHENATED_POSESSIVE_ALPHA'
#              'FLOAT_E', 'FLOAT_E', 'NONSPACE'):
#     name = 'RE_' + name
#     # this will hose up flake8
#     locals()[name + '_B'] = r'(' + locals()[name] + ')\\b'

# RE_CAMEL_CASE = ('(((' + RE_WORD_CAPITALIZED_B + ')|(' + RE_WORD_LOWERCASE + '))' + '(' + RE_ACRONYM + '))|' +
#                  '((' + RE_ACRONYM + '|' + RE_WORD_CAPITALIZED + '|' + RE_WORD_LOWERCASE + ')(' +
#                  RE_WORD_CAPITALIZED + ')+)' + r'\b')
# RE_CAMEL_CASE = CRE_CAMEL_CASE = re.compile(RE_CAMEL_CASE)

# always list RE's from most greedy to least greedy []+, []*, []?, then [], supersets before subsets in char groups []
RE_TOKEN = r'|'.join(['[.]' + RE_HYPHENATED_ALPHANUM, RE_HYPHENATED_ALPHA,
                      RE_HYPHENATED_ALPHANUM,
                      RE_UNHYPHENATED_CONTRACTED_ALPHA_B,
                      RE_USD_DECIMAL_BMK, RE_USD_BMK_B, RE_USD_CENTS, RE_USD,
                      RE_DECADE, RE_YEAR,
                      RE_ACRONYM,
                      RE_FLOAT_E, RE_FLOAT,
                      RE_NONWORD])
# more resrictive version that overrides the one above
RE_TOKEN = r'|'.join([RE_DOUBLEQUOTE,
                      RE_USD_DECIMAL_BMK_B, RE_USD_BMK_B, RE_USD_CENTS_B, RE_USD_B,
                      RE_DECADE_B, RE_YEAR_B,
                      RE_ACRONYM_B,
                      RE_FLOAT_E_B, RE_FLOAT_B,
                      '[.]' + RE_HYPHENATED_ALPHANUM_B,
                      RE_HYPHENATED_POSESSIVE_ALPHA_B,
                      RE_HYPHENATED_DOTTED_ALPHANUM_B,
                      # FIXME: Plural words at end single quotes around plural words to be interpretted as possessive
                      RE_POSESSIVE_ALPHA_B,
                      RE_HYPHENATED_ALPHA_B,
                      RE_HYPHENATED_ALPHANUM_B,
                      RE_UNHYPHENATED_CONTRACTED_ALPHA_B,
                      RE_NONWORD])

RE_FINDALL_TOKENS = r'\w+(?:\'\w+)?|[^\w\s]'

LIST_RE_TOKEN_NAMED = [
    'RE_USD_DECIMAL_BMK',
    'RE_USD_BMK',
    'RE_USD_CENTS',
    'RE_USD',
    'RE_DECADE',
    'RE_YEAR',
    'RE_ACRONYM',
    'RE_FLOAT_E',
    'RE_FLOAT',
    'RE_HYPHENATED_ALPHA',
    'RE_UNHYPHENATED_CONTRACTED_ALPHA',
    'RE_NONWORD']
RE_TOKEN_NAMED = r'|'.join([r'(?P<{}>{})'.format(s.lower()[3:], globals()[s]) for s in LIST_RE_TOKEN_NAMED])

CRE_WORD_DELIM = re.compile(RE_WORD_DELIM)
CRE_HTML_TAG = re.compile(RE_HTML_TAG)
CRE_TOKEN = re.compile(RE_TOKEN)

RE_BAD_FILENAME = '[{}]'.format(re.escape(string.punctuation.replace('-', '').replace('_', '') + string.unprintable))
CRE_BAD_FILENAME = re.compile(RE_BAD_FILENAME)
CRE_WHITESPACE = re.compile(r'\s')


#####################################################
# IDE and code refactoring regexes
# Tested in Sublime Text 2

# Find redefinitions of the same regex in the same file
RE_REDEF = r'(\n[C]?RE_[A-Z_]+[ ])[\w\W]*\1'
# IDE and code refactoring regexes
#####################################################

# pugnlp.regexes
#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################
# nlpia.regexes

# kind of like stopwords, but just the words that are commonly lowercased in article titles
TITLE_LOWERWORDS = sorted('of a in the on as if and or but with'.split())
RE_ACRONYM_IGNORE = '(?:' + '|'.join(TITLE_LOWERWORDS) + ')'

RE_BREAK_CHARCLASS = r'\b[^-_a-zA-Z0-9]'  # like \W but doesn't allow "-" to break words
RE_STYLEMARK = r'[_*+^~]'  # italics, bold, math, superscript, subscript

RE_BOLD_START = r'(?:(?<![*])\*(?=[a-zA-Z0-9]))'  # start delimiter for bolded word
RE_BOLD_END = r'(?:\*(?![*]))'  # end delimiter for bolded word
RE_BOLD_CHAR_START = r'(?:(?<![*])\*\*(?=[a-zA-Z0-9]))'  # start delimiter for single character bolded
RE_BOLD_CHAR_END = r'(?:(?=[a-zA-Z0-9])\*\*(?![*]))'  # end delimiter for single character bolded

RE_ITALIC_START = r'(?:(?<!_)_(?=[a-zA-Z0-9]))'  # start delimiter for italicized word
RE_ITALIC_END = r'(?:_(?!_))'  # end delimiter for italicized word
RE_ITALIC_CHAR_START = r'(?:(?<!_)__(?=[a-zA-Z0-9]))'  # start delimiter for single character italicized
RE_ITALIC_CHAR_END = r'(?:(?=[a-zA-Z0-9])__(?!_))'  # end delimiter for single character italicized

RE_WORD_CHARCLASS = r'[-a-zA-Z0-9]'  # like \w but for English, not code, so "-" allowed but not "_"
RE_OPTIONAL_WORD = '(?:' + RE_WORD_CHARCLASS + '{0,16})'  # like \w but for English, not code, so "-" allowed but not "_"
RE_ENGLISH_WORD = '(?:' + RE_WORD_CHARCLASS + '{1,16})'

RE_STYLE_START = '(?:' + '|'.join(
    [RE_BOLD_START, RE_BOLD_CHAR_START, RE_ITALIC_START, RE_ITALIC_CHAR_START]
) + ')'
RE_STYLE_END = '(?:' + '|'.join(
    [RE_BOLD_END, RE_BOLD_CHAR_END, RE_ITALIC_END, RE_ITALIC_CHAR_END]
) + ')'

PATTERNS = {
    'word': RE_ENGLISH_WORD, 'word0': RE_OPTIONAL_WORD,
    'boldstart': RE_BOLD_START, 'boldend': RE_BOLD_END,
    'boldcharstart': RE_BOLD_CHAR_START, 'boldcharend': RE_BOLD_CHAR_END,
    'italicstart': RE_ITALIC_START, 'italicend': RE_ITALIC_END,
    'italiccharstart': RE_ITALIC_CHAR_START, 'italiccharend': RE_ITALIC_CHAR_END,
    'stylestart': RE_STYLE_START, 'styleend': RE_STYLE_END,
}

PATTERNS.update({'stylestart': RE_STYLE_START, 'styleend': RE_STYLE_END})
CHARCLASSES = {'w': RE_WORD_CHARCLASS, 'W': RE_BREAK_CHARCLASS, 'b': RE_BREAK_CHARCLASS}
PATTERNS.update(CHARCLASSES)

RE_ACRONYM2 = '\\b(?P<s2>' \
    r'{stylestart}?([a-zA-Z]){styleend}?{word}{styleend}?{b}' \
    r'{stylestart}?([a-zA-Z]){styleend}?{word}{styleend}?' \
    ')[\\s]?[\\s]?\\((?P<a2>\2[-.*_]?[\\s]?\3[.]?)\\)'.format(**PATTERNS)
RE_ACRONYM3 = r'\b[_*]{0,2}(?P<s3>(\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16})[_*]{0,2}[ ]\((?P<a3>\6[-.*_ ]{0,2}\7[-.*_ ]{0,2}\8[-.*_ ]{0,2})\)'
RE_ACRONYM4 = r'\b[_*]{0,2}(?P<s4>(\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16})[_*]{0,2}[ ]' \
    r'\((?P<a4>\11[-.*_ ]{0,2}\12[-.*_ ]{0,2}\13[-.*_ ]{0,2}\14[-.*_ ]{0,2})\)'
RE_ACRONYM5 = r'\b[_*]{0,2}(?P<s5>(\w)[-\w0-9]{0,16}[ ](\w)[-\w0-9]{0,16}' \
    r'[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16}[ ](\w)[-*\w0-9]{0,16})' \
    r'[_*]{0,2}[ ]\((?P<a5>\17[-.*_ ]{0,2}\18[-.*_ ]{0,2}\19[-.*_ ]{0,2}\20[-.*_ ]{0,2}\21[-.*_ ]{0,2})\)'
CRE_ACRONYM = re.compile('|'.join((RE_ACRONYM2, RE_ACRONYM3, RE_ACRONYM4, RE_ACRONYM5)), re.IGNORECASE)

RE_ACRONYM2 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\2\3)\)'
RE_ACRONYM3 = r'((\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16}[ ](\w)[\w0-9]{2,16})[ ]\((\6\7\8)\)'
CRE_ACRONYM = re.compile(RE_ACRONYM2 + '|' + RE_ACRONYM3, re.IGNORECASE)

RE_TLD_POPULAR = 'com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me|dev|social|ai'
CRE_TLD_POPULAR = re.compile(RE_TLD_POPULAR)

RE_URL_CHARS = r'([^"\'`\)\(\[\]\s]*)'
#               r'(?P<url>(?P<scheme>(?P<scheme_type>http|ftp|https)://)?([^/:(\["\'`)\]\s]+[.])(com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me|dev|social|ai)([^"\'`)\]\s]*))'
RE_URL_POPULAR = RE_URL_SIMPLE = (
    r'(?P<url>(?P<scheme>(?P<scheme_type>http|ftp|https)://)?([^/:(\["\'`)\]\s]+[.])(' +
    RE_TLD_POPULAR + ')\b(?P<path>[/]+[^"\'`()\\]\\s]*)?)'
)
CRE_URL_SIMPLE = re.compile(RE_URL_SIMPLE)
RE_URL_POPULAR = RE_URL_SIMPLE = (
    r'(?P<url>(?P<scheme>(?P<scheme_type>http|ftp|https)://)?([^/:("\'\\`)\[\]\s]+[.])(?P<tld>com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me|dev|social|ai)\b(?P<path>[/]+[^"\'`)\]\s]*)?)'
)

CRE_URL_SIMPLE = re.compile(RE_URL_SIMPLE)
CRE_URL_POPULAR = re.compile(RE_URL_POPULAR)

RE_URL_WITH_SCHEME = RE_URL_SIMPLE.replace('://)', '://)?')  # require scheme
CRE_URL_WITH_SCHEME = re.compile(RE_URL_WITH_SCHEME)

RE_HYPERLINK = RE_URL_WITH_SCHEME + r'\[(?P<name>[^\]]+)\]'
CRE_HYPERLINK = re.compile(RE_HYPERLINK)
"""
>>> CRE_SLUG_DELIMITTER.sub('-', 'thisSlug-should|beHypenatedInLots_OfPlaces')
'this-Slug-should-be-Hypenated-In-Lots-Of-Places'
"""
CRE_SLUG_DELIMITTER = re.compile(r'[^a-zA-Z]+|(?<=[a-z])(?=[A-Z])')
"""
>>> CRE_FILENAME_EXT.search('~/.bashrc.asciidoc.ext.ps4.42').group()
'.asciidoc.ext.ps4.42'
>>> CRE_FILENAME_EXT.sub('', 'this/path/has/a/file.html')
'this/path/has/a/file'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc.ext.ps4.123').group()
'.asciidoc.ext.ps4.123'
>>> CRE_FILENAME_EXT.search('.bashrc..asciidoc..ext.ps4.123').group()
'.ext.ps4.123'
"""
CRE_FILENAME_EXT = re.compile(r'(?<=[.a-zA-Z0-9_])([.][a-zA-Z0-9]{1,8}){1,5}$')


def splitext(filepath):
    """ Like os.path.splitext except splits compound extensions as one long one

    >>> splitext('~/.bashrc.asciidoc.ext.ps4.42')
    ('~/.bashrc', '.asciidoc.ext.ps4.42')
    >>> splitext('~/.bash_profile')
    ('~/.bash_profile', '')
    """
    exts = getattr(CRE_FILENAME_EXT.search(filepath), 'group', str)()
    return (filepath[:(-len(exts) or None)], exts)


# ? \(\): ()
# \': '"'"'
# \s: [:space:]
# RE_URL_BASH_ESCAPE = '((http|ftp|https)://)?[^/:\(\[\"'"'"'\`\)\] \t\n]+[.](com|org|edu|gov|net|mil|uk|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|io|me)[^\"'"'"'\`\)\] \t\n]*'  # noqa


def to_tsv():
    """ Save all regular expressions to a tsv file so they can be more easily copy/pasted into Sublime """
    with open(os.path.join(DATA_DIR, 'regexes.tsv'), mode='wt') as fout:
        vars = copy.copy(tuple(globals().items()))
        for k, v in vars:
            if k.lower().startswith('cre_'):
                fout.write(k[4:] + '\t' + v.pattern + '\n')
            elif k.lower().startswith('re_'):
                fout.write(k[3:] + '\t' + v.pattern + '\n')


class Pattern:
    """ Container for _regex.Pattern object augmented with Irregular matching rules

    >>> pattern = Pattern('Aaron[ ]Swartz')
    >>> pattern.match('Aaron Swartz').group()
    'Aaron Swartz'
    >>> pattern.fullmatch('Aaron Swartz!!')
    >>> pattern.match('Aaron Swartz!!').group()
    'Aaron Swartz'
    """

    def __init__(self, pattern):
        pattern = getattr(pattern, 'pattern', pattern)
        self._compiled_pattern = pattern if hasattr(pattern, 'pattern') else re.compile(pattern)
        self._cre = self._compiled_pattern
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__'):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                log.debug('{}.{}.Pattern successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, name, '()' if callable(attr) else ''))
            except:  # noqa
                log.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))


class REPattern:
    """ Container for re.SRE_Pattern object augmented with Irregular matching rules

    >>> pattern = REPattern('Aaron[ ]Swartz')
    >>> pattern.match('Aaron Swartz').group()
    'Aaron Swartz'
    >>> pattern.fullmatch('Aaron Swartz!!')
    >>> pattern.fullmatch('Aaron Swartz').group()
    'Aaron Swartz'
    >>> pattern.match('Aaron Swartz!!').group()
    'Aaron Swartz'
    """

    def __init__(self, pattern):
        self._compiled_pattern = re.compile(pattern)
        for name in dir(self._compiled_pattern):
            if name in ('__class__', '__init__', 'fullmatch') and getattr(self, name, None):
                continue
            attr = getattr(self._compiled_pattern, name)
            try:
                setattr(self, name, attr)
                log.debug('{}.{}.{} successfully "inherited" `_regex.Pattern.{}{}`'.format(
                    __package__, __name__, self.__class__, name, '()' if callable(attr) else ''))
            except:  # noqa
                log.warning('Unable to "inherit" `_regex.Pattern.{}{}`'.format(
                    name, '()' if callable(attr) else ''))

    def fullmatch(self, *args, **kwargs):
        return re.fullmatch(self._compiled_pattern.pattern, *args, **kwargs)

# nlpia.regexes
#####################################################################################
#####################################################################################
