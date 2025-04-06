# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt


num_examples = 7
seq_len = 5
embedding_size = 1

dataset = torch.arange(
    num_examples * seq_len * embedding_size,
    dtype=torch.float)
dataset.resize_(num_examples, seq_len, embedding_size)

df = pd.DataFrame(np.arange(
    num_examples * seq_len * embedding_size,
    dtype=float).reshape(num_examples, seq_len * embedding_size))
IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'

dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
dataset = torch.from_numpy(df.values).resize(num_examples, seq_len, embedding_size)

x = dataset[0]
x.resize_(seq_len, embedding_size)

conv = nn.Conv1d(in_channels=embedding_size, out_channels=1, groups=None, stride=1, kernel_size=2)
print(conv.weight)
hist
num_examples = 7
seq_len = 5
embedding_size = 1

dataset = torch.arange(
    num_examples * seq_len * embedding_size,
    dtype=torch.float)
dataset.resize_(num_examples, seq_len, embedding_size)

df = pd.DataFrame(np.arange(
    num_examples * seq_len * embedding_size,
    dtype=float).reshape(num_examples, seq_len * embedding_size))
IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'

dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
dataset = torch.from_numpy(df.values).resize(num_examples, seq_len, embedding_size)

x = dataset[0]
x.resize_(seq_len, embedding_size)
conv = nn.Conv1d(
    in_channels=embedding_size,
    out_channels=1,
    # groups=None,
    stride=1,
    kernel_size=2
)
print(conv.weight)
conv.state_dict()
state = conv.state_dict()
state['bias'] = torch.tensor([0.1])
conv.load_state_dict(state)
state['weight'] = torch.tensor([[[1., 2.]]])
state['weight'] = torch.tensor([[[-1., -2.]]])
x
conv.forward(x)
x.size()
x.resize_([1, 5, 1])
conv.forward(x)
x.size()
x.resize_([1, 1, 5])
x.size()
conv.forward(x)
conv = nn.Conv1d(
    in_channels=embedding_size,
    out_channels=1,
    # groups=None,
    stride=1,
    kernel_size=2
)
print(conv.weight)
conv.weight.dtype
conv.bias.dtype
x.dtype
data = np.arange(
    num_examples * seq_len * embedding_size,
    dtype=np.float32,
)
data = data.reshape(num_examples, seq_len * embedding_size)
df = pd.DataFrame(data)

IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'
dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)

dataset = torch.from_numpy(df.values)
dataset.resize_(num_examples, seq_len, embedding_size)

x = dataset[0]
x.resize_(seq_len, embedding_size)

conv = nn.Conv1d(
    in_channels=embedding_size,
    out_channels=1,
    # groups=None,
    stride=1,
    kernel_size=2
)
print(conv.weight)

conv.forward(x)
x.resize_(num_channels, embedding_size, seq_len)
num_channels = 1
x.resize_(num_channels, embedding_size, seq_len)
conv.forward(x)
state = conv.state_dict()
state['weight'] = torch.tensor(np.array([[[-1., -2.]]], dtype=np.float32))
conv.load_state_dict(state)
conv.forward(x)
state['bias'] = torch.tensor([0.1])
conv.forward(x)
conv.state_dict()
state['bias'] = torch.tensor([0.1])
conv.load_state_dict(state)
conv.forward(x)
state['weight'] = torch.tensor(np.array([[[-2, 3]]], dtype=np.float32))
state['bias'] = torch.tensor([0])
conv.load_state_dict(state)
conv.forward(x)
x
hist
%run minimalcnn
x
%run minimalcnn
%run minimalcnn
%run minimalcnn
nn.MaxPool1d(kernel_len, stride)
nn.MaxPool1d(kernel_size, stride)
pool = nn.MaxPool1d(kernel_size, stride)
pool.forward(y)
pool = nn.MaxPool1d(pool_size, pool_stride)
pool_size = 3
pool_stride = 2
pool = nn.MaxPool1d(pool_size, pool_stride)
pool.forward(y)
hist
import spacy
nlp = spacy.load('en_core_web_md')
spacy.cli.download('en_core_web_md')
nlp = spacy.load('en_core_web_md')
from nlpia2.init import maybe_download, DATA_DIR, HOME_DATA_DIR
quotes = maybe_download(filename='quotes.yml')
for q in quotes:
    print(q['text'])
quotes = yaml.full_load(maybe_download(filename='quotes.yml').open())
import yaml
quotes = yaml.full_load(maybe_download(filename='quotes.yml').open())
for q in quotes:
    print(q['text'])
for q in quotes:
    print([t.pos_ for t in q['text']])
for q in quotes:
    print([t.pos_ for t in nlp(q['text'])])
df = []
for q in quotes:
    df.append([t.pos_ for t in nlp(q['text'])])
df[0:2]
df = []
for q in quotes:
    df.append([1 if t.pos_ == 'ADJ' else 0 for t in nlp(q['text'])])
df = []
for q in quotes:
    text = q['text']
    df.append([t.pos_ for t in nlp(text)])
    df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
df = pd.DataFrame(df)
df
df.T.sum()
df.T.sum() / df.T.apply(len)
df = []
adverby_quotes = []
for i, q in enumerate(quotes):
    text = q['text']
    df.append([t.pos_ for t in nlp(text)])
    df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
    bits = df[-1]
    if (1, 1) in zip(bits[:-1], bits[1:]):
        adverby_quotes.append(i)
df.iloc[adverby_quotes]
df = []
adverby_quotes = []
for i, q in enumerate(quotes):
    text = q['text']
    df.append([t.pos_ for t in nlp(text)])
    df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
    bits = df[-1]
    if (1, 1) in zip(bits[:-1], bits[1:]):
        adverby_quotes.append(i)
np.array(df)[adverby_quotes]
df = []
adverby_quotes = []
for i, q in enumerate(quotes):
    text = q['text']
    df.append([t.pos_ for t in nlp(text)])
    df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
    bits = df[-1]
    if (1, 1) in zip(bits[:-1], bits[1:]):
        adverby_quotes.append(i)
df = pd.DataFrame(df).iloc[adverby_quotes]
(df.iloc[0] > -1).sum()
(df.iloc[1] > -1).sum()
df['len'] = (df > -1).sum(axis=1)
df.len
df.sort_values('len')
len(df)
df.sort_values('len')[:5]
np.array(quotes)[df.sort_values('len')[:5].index.values]
hist - o - p - f find - short - adverby - quote.yml
hist - o - p - f find - short - adverby - quote.hist.md
hist - f find - short - adverby - quote.hist.py
hist
ls - hal
rm find - short - adverby - quote.yml
more find - short - adverby - quote.hist.py
more find - short - adverby - quote.hist.md
df
df.index[3]
pd.options.display.max_rows = 7
pd.options.display.max_cols = 7
pd.options.display.max_columns = 7
df
df = df.sort_values('len')
shortest5 = df[:5].index.values]
shortest5 = df[: 5].index.values
shortest5
quotes[92]
hist - o - p - f 'find-short-adverby-quote-and-minimalcnn.hist.md'
hist - f 'find-short-adverby-quote-and-minimalcnn.hist.py'
more 'find-short-adverby-quote-and-minimalcnn.hist.py'
ls find *
hist - o - p - f find - short - adverby - quote - minimalcnn.hist.md
hist - f find - short - adverby - quote - minimalcnn.hist.py
ls
ls find *
rm find - short - adverby - quote.hist.md
rm find - short - adverby - quote.hist.py
more find - short - adverby - quote - minimalcnn.hist.md
quotes[92]
df.iloc[92]
df.loc[92]
df.loc[92].dropna()
df.loc[92].dropna()[-5: ]
df.loc[92].dropna()[-7: ]
list(df.loc[92].dropna())
quote= quotes[92]
doc= nlp(quote['text'])
quote['tags']
[(t.text, t.pos_, t.pos_ == 'ADV') for t in doc]
tagged_quote= [(t.text, t.pos_, int(t.pos_ == 'ADV')) for t in doc]
tagged_quote
tagged_quote= [(int(t.pos_ == 'ADV'), t.pos_, t.text) for t in doc]
tagged_quote
pd.DataFrame(tagged_quote)
pd.options.display.max_rows= 21
pd.DataFrame(tagged_quote)
pd.DataFrame(tagged_quote, columns='is_adv pos word'.split())
pd.DataFrame(tagged_quote, columns='is_adv POS word'.split())
hist - f find - short - adverby - quote - minimalcnn.hist.py
