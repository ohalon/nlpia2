>>> import re
>>> import regex
>>> type(pattern)
>>> pattern = '[0-9]'
>>> re.match(pattern, '7')
<re.Match object; span=(0, 1), match='7'>
>>> re.match(pattern, '-1')
>>> re.match(pattern, 'x')
>>> re.match(pattern, 'a')
>>> re.match(pattern, '42')
<re.Match object; span=(0, 1), match='4'>
>>> pattern = '[0-9]*'
>>> re.match(pattern, '42')
<re.Match object; span=(0, 2), match='42'>
>>> re.match(pattern, 'x')
<re.Match object; span=(0, 0), match=''>
>>> pattern = '[0-9]+'
>>> re.match(pattern, '42')
<re.Match object; span=(0, 2), match='42'>
>>> pattern = '[0-9]{3}'
>>> re.match(pattern, '42')
>>> re.match(pattern, '420')
<re.Match object; span=(0, 3), match='420'>
>>> pattern = '([0-9]{1,3}\\.){2}'
>>> re.match(pattern, '4.2.')
<re.Match object; span=(0, 4), match='4.2.'>
>>> pattern = '([0-9]{1,3}\\.){3}([0-9]{1,3})'
>>> re.match(pattern, '123.45.1.1')
<re.Match object; span=(0, 10), match='123.45.1.1'>
>>> re.match(pattern, 'garbage 123.45.1.1')
>>> re.match(pattern, 'localhost=127.0.0.1')
>>> re.match(pattern, 'localhost=127.0.0.1?')
>>> pattern = '[^0-9]*([0-9]{1,3}\\.){3}([0-9]{1,3}).*'
>>> re.match(pattern, 'localhost=127.0.0.1?')
<re.Match object; span=(0, 20), match='localhost=127.0.0.1?'>
>>> re.match(pattern, 'localhost=127.0.0.1.')
<re.Match object; span=(0, 20), match='localhost=127.0.0.1.'>
>>> match = re.match(pattern, 'localhost=127.0.0.1.')
>>> match.end()
20
>>> match.start()
0
>>> match.groups()
('0.', '1')
>>> pattern = '[^0-9]*([0-9]{1,3}\\.){3}([0-9]{1,3})[^0-9]*'
>>> match = re.match(pattern, 'localhost=127.0.0.1.')
>>> match.groups()
('0.', '1')
>>> match.groups()[0]
'0.'
>>> match.groups()[1]
'1'
>>> pattern = '[^0-9]*((([0-9]{1,3}\\.){3})([0-9]{1,3}))[^0-9]*'
>>> match = re.match(pattern, 'localhost=127.0.0.1.')
>>> match.groups()[1]
'127.0.0.'
>>> match.groups()
('127.0.0.1', '127.0.0.', '0.', '1')
>>> match.groups()[0]
'127.0.0.1'
>>> match
<re.Match object; span=(0, 20), match='localhost=127.0.0.1.'>
>>> pattern = '((([0-9]{1,3}\\.){3})([0-9]{1,3}))'
>>> match = re.match(pattern, 'localhost=127.0.0.1.')
>>> matches = re.search(pattern, 'localhost=127.0.0.1.')
>>> matches
<re.Match object; span=(10, 19), match='127.0.0.1'>
>>> matches = re.search(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
<re.Match object; span=(10, 19), match='127.0.0.1'>
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
[('127.0.0.1', '127.0.0.', '0.', '1'), ('254.0.0.5', '254.0.0.', '0.', '5')]
>>> matches = re.finditer(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
<callable_iterator at 0x7f70d530fd90>
>>> matches = re.finditer('xyz', 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
<callable_iterator at 0x7f70d55e4b20>
>>> matches = re.finditer(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> iter
<function iter>
>>> iter('abc')
<str_iterator at 0x7f70d51845b0>
>>> for x in iter('abc'):
...     print(x)
...
>>> g = iter('abc')
>>> for x in g:
...     print(x)
...
>>> for x in g:
...     print(x)
...
>>> for x in g():
...     print(x)
...
>>> dir(g)
['__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__length_hint__',
 '__lt__',
 '__ne__',
 '__new__',
 '__next__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__setstate__',
 '__sizeof__',
 '__str__',
 '__subclasshook__']
>>> next(g)
>>> g = iter('abc')
>>> next(g)
'a'
>>> next(g)
'b'
>>> next(g)
'c'
>>> next(g)
>>> matches
<callable_iterator at 0x7f70d529fd90>
>>> for m in matches:
...     print(m)
...
>>> re.compile(pattern)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.UNICODE)
>>> cre = re.compile(pattern)
>>> cre
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE | re.UNICODE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.IGNORECASE|re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE and re.UNICODE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE or re.UNICODE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.IGNORECASE|re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE & re.UNICODE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE | re.UNICODE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.IGNORECASE|re.UNICODE)
>>> re.compile(pattern, re.IGNORECASE)
re.compile(r'((([0-9]{1,3}\.){3})([0-9]{1,3}))', re.IGNORECASE|re.UNICODE)
>>> pattern = r'[-a-zA-Z]+'
>>> re.compile(pattern)
re.compile(r'[-a-zA-Z]+', re.UNICODE)
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', 'and']
>>> pattern = r'[-a-z0-9]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '127', '0', '0', '1', 'and', '254', '0', '0', '5']
>>> pattern = r'[-a-z0-9.=]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost=127.0.0.1.', 'and', '254.0.0.5']
>>> pattern = r'[-a-z0-9]+|[.-?!&%]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '=127.0.0.1.', 'and', '254', '.0.0.5']
>>> pattern = r'[-a-z0-9]+|[$0-9.]+|[.-?!&%]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '=127.0.0.1.', 'and', '254', '.0.0.5']
>>> pattern = r'[-a-z0-9]+|[$0-9.]+|[.-?!&%=]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '=127.0.0.1.', 'and', '254', '.0.0.5']
>>> pattern = r'[$0-9.]+|[-a-z0-9]+|[.-?!&%=]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '=127.0.0.1.', 'and', '254.0.0.5']
>>> pattern = r'[=]+|[$0-9.]+|[-a-z0-9]+|[.-?!&%=]+'
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. and 254.0.0.5')
>>> matches
['localhost', '=', '127.0.0.1.', 'and', '254.0.0.5']
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. AND 254.0.0.5')
>>> matches
['localhost', '=', '127.0.0.1.', '254.0.0.5']
>>> matches = re.findall(pattern, 'localhost=127.0.0.1. AND 254.0.0.5', re.IGNORECASE)
>>> matches
['localhost', '=', '127.0.0.1.', 'AND', '254.0.0.5']
>>> history -o -p -f ch01_re_tokenizers.md
>>> history -o -p -f ch01_re_tokenizers.hist.md
