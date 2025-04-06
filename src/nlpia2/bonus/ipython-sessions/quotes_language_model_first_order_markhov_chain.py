import yaml
yaml.full_load(open('data/quotes.yml'))
quotes = _
for q in quotes:
    print(q['text'])
sentences = []
for q in quotes:

    print(q['text'])
import spacy
nlp = spacy.load('en_core_web_md')
sentences = []
for q in quotes:

    txt = q['text']
    doc = nlp(txt)
    sentences.extend([s.text for s in doc.sents])
sentences
len(sentences)
import numpy as np
np.random.choose(sentences)
dir(np.random)
np.random.choice(sentences)
np.random.choice(sentences)
np.random.choice(sentences)
np.random.choice(sentences)
np.random.choice(sentences)
np.random.choice(sentences)
np.random.choice(sentences, prob=[1] + [0] * (len(sentences) - 1))
np.random.choice(sentences, probs=[1] + [0] * (len(sentences) - 1))
help(np.random.choice)
np.random.choice(sentences, p=[1] + [0] * (len(sentences) - 1))
sentences[0]
from collections import Counter
Counter(sentences[0])
sentences = []
for q in quotes:

    txt = q['text']
    doc = nlp(txt)
    sentences.extend([s.text for s in doc.sents])
sentences = [re.find_all(r'[a-z ]', s.lower()) for s in sentences]
import re
sentences = [re.find_all(r'[a-z ]', s.lower()) for s in sentences]
sentences = [re.findall(r'[a-z ]', s.lower()) for s in sentences]
sentences[0]
sentences = [['<SOS>'] + clist + ['<EOS>'] for clist in sentences]
sentences = [Counter(clist) for clist in sentences]
sentences[0]
sum(sentences)
type(sentences[0])
import pandas as pd
sentences[1]
sum(sentences[:2])
df = pd.DataFrame(sentences)
df.sum()
df.head()
prob = df.sum()
prob = prob / prob.sum()
prob
generated_sentence = []
while c != '<EOS>':
    c = np.random.choice(prob.index.values, p=prob.values)
    generated_sentence.append(c)
generated_sentence = []
c = '<SOS>'
while c != '<EOS>':
    c = np.random.choice(prob.index.values, p=prob.values)
    generated_sentence.append(c)
generated_sentence
''.join(generated_sentence)
generated_sentence = []
c = '<SOS>'
while c != '<EOS>':
    c = np.random.choice(prob.index.values, p=prob.values)
    generated_sentence.append(c)
''.join(generated_sentence)
generated_sentence = []
c = '<SOS>'
while c != '<EOS>':
    c = np.random.choice(prob.index.values, p=prob.values)
    generated_sentence.append(c)
''.join(generated_sentence)
generated_sentence = []
c = '<SOS>'
while c != '<EOS>':
    while c != '<SOS>':
        c = np.random.choice(prob.index.values, p=prob.values)
    generated_sentence.append(c)
generated_sentence = []
c = '<SOS>'
while c != '<EOS>':
    c = np.random.choice(prob.index.values, p=prob.values)
    if c == '<SOS>':
        continue
    generated_sentence.append(c)
generated_sentence
''.join(generated_sentence)


def generate_sentence(model_order=0):
    generated_sentence = []
    c = '<SOS>'
    while c != '<EOS>':
        c = np.random.choice(prob.index.values, p=prob.values)
        if c == '<SOS>':
            continue
        generated_sentence.append(c)
    return ''.join(generated_sentence)


generate_sentence()
generate_sentence()
generate_sentence()
generate_sentence()
hist


def generate_ngrams(seq):
    return list(zip(seq[:-1], seq[1:]))


generate_ngrams('Hello world')
hist
sentences
df_counts = pd.DataFrame(sentences)
df.head()

sentences = []
for q in quotes:

    txt = q['text']
    doc = nlp(txt)
    sentences.extend([s.text for s in doc.sents])
[generate_ngrams(s) for s in sentences]
sentences = [re.findall(r'[a-z ]', s.lower()) for s in sentences]
sentences = [['<SOS>'] + clist + ['<EOS>'] for clist in sentences]
sentence_2grams = list(generate_ngrams(s) for s in sentences)
sentence_2grams[0]
counts = {}
counts = dict(zip(probs.index.values, [] * len(probs)))
for s in sentence_2grams:
    for tg in s:
        counts[tg[0]] += [tg[1]]
counts = dict(zip(probs.index.values, [] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        counts[tg[0]] += [tg[1]]
counts = dict(zip(prob.index.values, [] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        counts[tg[0]] += [tg[1]]
prob.index.values
counts
counts = dict(zip(prob.index.values, [[]] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        counts[tg[0]] += [tg[1]]
counts
counts = dict(zip(prob.index.values, [[]] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        counts[tg[0]] += [tg[1]]
counts['s']
counts['s']
len(counts['s'])
len(counts['e'])
counts = dict(zip(prob.index.values, [[]] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        print tg[0], tg[1]
        counts[tg[0]] += [tg[1]]
        break
counts = dict(zip(prob.index.values, [[]] * len(prob)))
for s in sentence_2grams:
    for tg in s:
        print(tg[0], tg[1])
        counts[tg[0]] += [tg[1]]
        break
counts = dict(zip(prob.index.values, [[]] * len(prob)))
for s in sentence_2grams:
    print(s)
    for tg in s:
        print(tg[0], tg[1])
        counts[tg[0]] += [tg[1]]
        break
counts = {}
for c in prob.index.values:
    counts[c] = list()
for s in sentence_2grams:
    print(s)
    for tg in s:
        print(tg[0], tg[1])
        counts[tg[0]].append(tg[1])
        break
counts
counts['SOS']
counts['<SOS>']
counters = dict()
for k, v in counts.items():
    counters[k] = Counter(v)
dfcounters = pd.DataFrame(counters)
dfcounters.head()
counts['<EOS>']
counts['t']
counts = {}
for c in prob.index.values:
    counts[c] = list()
for s in sentence_2grams:
    print(s)
    for tg in s:
        print(tg[0], tg[1])
        counts[tg[0]].append(tg[1])
counts['t']
counters = dict()
for k, v in counts.items():
    counters[k] = Counter(v)
dfcounters = pd.DataFrame(counters)
dfcounters.head()
dfcounters / dfcounters.sum()
dfcounters = dfcounters.fillna(0)
dfcounters['<EOS>']
dfcounters['<EOS>'].sum()
dfcounters['<EOS>'][' '] = 1
dfcounters.sum()
dfcounters / dfcounters.sum()
dfprobs = dfcounters / dfcounters.sum()
dfprobs.head()
import pandas as pd
from collections import Counter
import numpy as np


def generate_ngrams(seq):
    return list(zip([seq[:-1], seq[1:]]))


# def generate_ngrams(seq, n=2):
#     return list(zip([seq[i:-j] for i,j in zip(range(n-1), range(1,n))], seq[(n-1):]]))


def generate_sentence(model_order=0, probs=None):
    model_order = 0 if len(probs.shape) == 1 else 1
    # if model_order:
    #     model_order = len(probs.shape) - 1
    if model_order:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            ctemp = np.random.choice(probs.index.values, p=probs[c].values)
            if ctemp == '<SOS>':
                continue
            c = ctemp
            generated_sentence.append(c)
        return ''.join(generated_sentence)
    else:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            c = np.random.choice(probs.index.values, p=probs.values)
            if c == '<SOS>':
                continue
            generated_sentence.append(c)
        return ''.join(generated_sentence)


probs.head()
dfprobs.head()
generate_sentence(model_order=1, probs=dfprobs)
generate_sentence(model_order=1, probs=dfprobs)
generate_sentence(model_order=1, probs=dfprobs)
import pandas as pd
from collections import Counter
import numpy as np


def generate_ngrams(seq):
    return list(zip([seq[:-1], seq[1:]]))


# def generate_ngrams(seq, n=2):
#     return list(zip([seq[i:-j] for i,j in zip(range(n-1), range(1,n))], seq[(n-1):]]))


def generate_sentence(model_order=0, probs=None):
    model_order = 0 if len(probs.shape) == 1 else 1
    # if model_order:
    #     model_order = len(probs.shape) - 1
    if model_order:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            ctemp = np.random.choice(probs.index.values, p=probs[c].values)
            if ctemp == '<SOS>':
                continue
            print(ctemp)
            c = ctemp
            generated_sentence.append(c)
        return ''.join(generated_sentence)
    else:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            c = np.random.choice(probs.index.values, p=probs.values)
            if c == '<SOS>':
                continue
            generated_sentence.append(c)
        return ''.join(generated_sentence)


generate_sentence(model_order=1, probs=dfprobs)
hist - o - p - f quotes_language_model_firt_order_markhov_chain.ipy
hist - f quotes_language_model_firt_order_markhov_chain.py
