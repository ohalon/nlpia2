import re
import regex
type(pattern)
pattern = '[0-9]'
re.match(pattern, '7')
re.match(pattern, '-1')
re.match(pattern, 'x')
re.match(pattern, 'a')
re.match(pattern, '42')
pattern = '[0-9]*'
re.match(pattern, '42')
re.match(pattern, 'x')
pattern = '[0-9]+'
re.match(pattern, '42')
pattern = '[0-9]{3}'
re.match(pattern, '42')
re.match(pattern, '420')
pattern = '([0-9]{1,3}\\.){2}'
re.match(pattern, '4.2.')
pattern = '([0-9]{1,3}\\.){3}([0-9]{1,3})'
re.match(pattern, '123.45.1.1')
re.match(pattern, 'garbage 123.45.1.1')
re.match(pattern, 'localhost=127.0.0.1')
re.match(pattern, 'localhost=127.0.0.1?')
pattern = '[^0-9]*([0-9]{1,3}\\.){3}([0-9]{1,3}).*'
re.match(pattern, 'localhost=127.0.0.1?')
re.match(pattern, 'localhost=127.0.0.1.')
match = re.match(pattern, 'localhost=127.0.0.1.')
match.end()
match.start()
match.groups()
pattern = '[^0-9]*([0-9]{1,3}\\.){3}([0-9]{1,3})[^0-9]*'
match = re.match(pattern, 'localhost=127.0.0.1.')
match.groups()
match.groups()[0]
match.groups()[1]
pattern = '[^0-9]*((([0-9]{1,3}\\.){3})([0-9]{1,3}))[^0-9]*'
match = re.match(pattern, 'localhost=127.0.0.1.')
match.groups()[1]
match.groups()
match.groups()[0]
match
pattern = '((([0-9]{1,3}\\.){3})([0-9]{1,3}))'
match = re.match(pattern, 'localhost=127.0.0.1.')
matches = re.search(pattern, 'localhost=127.0.0.1.')
matches
matches = re.search(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
matches = re.finditer(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
matches = re.finditer('xyz', 'localhost=127.0.0.1. and 254.0.0.5')
matches
matches = re.finditer(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
iter
iter('abc')
for x in iter('abc'):
    print(x)
g = iter('abc')
for x in g:
    print(x)
for x in g:
    print(x)
for x in g():
    print(x)
dir(g)
next(g)
g = iter('abc')
next(g)
next(g)
next(g)
next(g)
matches
for m in matches:
    print(m)
re.compile(pattern)
cre = re.compile(pattern)
cre
re.compile(pattern, re.IGNORECASE | re.UNICODE)
re.compile(pattern, re.IGNORECASE and re.UNICODE)
re.compile(pattern, re.IGNORECASE or re.UNICODE)
re.compile(pattern, re.IGNORECASE & re.UNICODE)
re.compile(pattern, re.IGNORECASE | re.UNICODE)
re.compile(pattern, re.IGNORECASE)
pattern = r'[-a-zA-Z]+'
re.compile(pattern)
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[-a-z0-9]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[-a-z0-9.=]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[-a-z0-9]+|[.-?!&%]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[-a-z0-9]+|[$0-9.]+|[.-?!&%]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[-a-z0-9]+|[$0-9.]+|[.-?!&%=]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[$0-9.]+|[-a-z0-9]+|[.-?!&%=]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
pattern = r'[=]+|[$0-9.]+|[-a-z0-9]+|[.-?!&%=]+'
matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
matches
matches = re.findall(pattern, 'localhost=127.0.0.1. AND 254.0.0.5')
matches
matches = re.findall(pattern, 'localhost=127.0.0.1. AND 254.0.0.5', re.IGNORECASE)
matches
history -o -p -f ch01_re_tokenizers.md
history -o -p -f ch01_re_tokenizers.hist.md
history -f ch01_re_tokenizers.hist.py
