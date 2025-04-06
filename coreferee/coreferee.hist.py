import spacy
spacy.__version__
import coreferee, spacy
nlp = spacy.load('en_core_web_trf')
python -m spacy validate
spacy.cli.validate
spacy.cli.validate()
nlp.add_pipe('coreferee')
doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
doc._.coref_chains
doc._.coref_chains.print()
doc._.coref_chains[0].print()
doc._.coref_chains[0]
doc._.coref_chains[0].pretty_representation
doc._.coref_chains[1].pretty_representation
doc
str(doc._.coref_chains[1])
doc._.coref_chains[1].repr()
repr(doc._.coref_chains[1])
doc._.coref_chains[2].pretty_representation
for i, tok in enumerate(doc):
    print(f'{i}:{tok}', end='')
tok
dir(tok)
for i, tok in enumerate(doc):
    print(f'{i}:{tok}{tok.whitespace_}', end='')
i = 0
for sent in doc.sents:
    for tok in sent:
        print(f'{i}:{tok}{tok.whitespace_}', end='')
        i += 1
hist
i = 0
headers = []
sents = []
for sent in doc.sents:
    headers.append('')
    sents.append('')
    for t in sent:
        tok = f'{t}{t.whitespace_}'
        idx = f'{idx}'
        if len(idx) >= len(tok):
            idx = ''
        else:
            idx += ' ' * (len(tok) - len(idx))
        headers[-1] += idx
        sents[-1] += tok
        i += 1
    print(headers[-1])
    print(sents[-1])
i = 0
headers = []
sents = []
for sent in doc.sents:
    headers.append('')
    sents.append('')
    for t in sent:
        tok = t.text + t.whitespace_
        idx = str(i)
        if len(idx) >= len(tok):
            idx = ''
        else:
            idx += ' ' * (len(tok) - len(idx))
        headers[-1] += idx
        sents[-1] += tok
        i += 1
    print(headers[-1])
    print(sents[-1])
i = 0
headers = []
sents = []
for sent in doc.sents:
    headers.append('')
    sents.append('')
    for t in sent:
        tok = t.text + t.whitespace_
        idx = str(i)
        if len(idx) >= len(tok):
            idx = ''
        else:
            idx += ' ' * (len(tok) - len(idx))
        headers[-1] += idx
        sents[-1] += tok
        i += 1
    print(headers[-1])
    print(sents[-1])
    print()
from spacy.cli import download
hist -o -p -f coreferee.hist.ipy
hist -f coreferee.hist.py
