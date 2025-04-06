```
$ pip install virtualenv
$ python -m virtualenv .venv
$ source .venv/bin/activate
```

```
$ pip install coreferee ipython jupyter  # <1>
$ python -c 'import spacy; print(spacy.__version__)'
```

```python
python -m spacy download en_core_web_lg  # <1>
python -m spacy download en_core_web_trf  # <2>
python -m coreferee install en
```

```python
>>> !pip install nlpia2_wikipedia  # <1>
>>> import wikipedia as wiki
>>> page = wiki.page('Timnit Gebru')
>>> text = page.content
```

```python
>>> import spacy, coreferee  # <1>
>>> nlptrf = spacy.load('en_core_web_trf')
>>> nlptrf.add_pipe('coreferee')
<coreferee.manager.CorefereeBroker at 0x...>
>>> doc_gebru = nlptrf(text_gebru)  # <2>
>>> doc_gebru._.coref_chains
[0: [13], [16], [26], [34],
 1: [51], [56]]
>>> doc_gebru._.coref_chains.print()
0: Gebru(13), she(16), she(26), she(34)  # <3>
1: advice(51), it(56)
```

```python
>>> def stringify_coreferences(doc):
...     i, headers, sents = 0, [], []
...     for sent in doc.sents:
...         headers.append('')
...         sents.append('')
...         for t in sent:
...             tok = t.text + t.whitespace_
...             idx = str(i)
...             if len(idx) >= len(tok):
...                 idx = ' '*len(tok)
...             else:
...                 idx += ' ' * (len(tok) - len(idx))
...             headers[-1] += idx
...             sents[-1] += tok
...             i += 1
...     return headers, sents
...     
```

```python
>>> def wrap_header_strings(headers, sents, width=70):
...     lines = []
...     for h, s in zip(headers, sents):
...         i = 0
...         while i < len(s):
...             i += width
...             lines.append(h[i-width:i])
...             lines.append(s[i-width:i])
...             lines.append('')
```

```python
>>> headers, sents = stringify_coreferences(doc_gebru)
>>> print('\n'.join(wrap_header_strings(headers, sents))
0  1 2   4    5    6    7  8  9        10            11    13    14
In a six-page mail sent to an internal collaboration list, Gebru descr

     15  16  17  18       19   21      22 23    24     25    26  27  2
ibes how she was summoned to a meeting at short notice where she was a

8    29 30       31  32    33  34  35        36 37   38  39    40  41
sked to withdraw the paper and she requested to know the names and rea

     42 43       44  45   46   47        49    50   51     52  53  54
sons of everyone who made that decision, along with advice for how to

55     56 57 58    59 60
revise it to Google's liking

>>> doc_gebru._.coref_chains.print()
0: Gebru(13), she(16), she(26), she(34)  # <3>
1: advice(51), it(56)
```