quote = "no word was ever as effective as a rightly timed pause."
nlp = spacy.load('en_core_web_md')

tagged_quote = [
    (int(t.pos_ == 'ADV'), t.pos_, t.text)  # <1>
    for t in nlp(quote)]

df_quote = pd.DataFrame(
    tagged_quote,
    columns='is_adv part_of_speech token'.split())

pd.options.display.max_rows = 20
print(df_quote)
import spacy
quote = "no word was ever as effective as a rightly timed pause."
nlp = spacy.load('en_core_web_md')

tagged_quote = [
    (int(t.pos_ == 'ADV'), t.pos_, t.text)  # <1>
    for t in nlp(quote)]

df_quote = pd.DataFrame(
    tagged_quote,
    columns='is_adv part_of_speech token'.split())

pd.options.display.max_rows = 20
print(df_quote)
import pandas as pd
quote = "no word was ever as effective as a rightly timed pause."
nlp = spacy.load('en_core_web_md')

tagged_quote = [
    (int(t.pos_ == 'ADV'), t.pos_, t.text)  # <1>
    for t in nlp(quote)]

df_quote = pd.DataFrame(
    tagged_quote,
    columns='is_adv part_of_speech token'.split())

pd.options.display.max_rows = 20
print(df_quote)
df_quote.T
quote = 'The right word may be effective, but no word was ever as effective as a rightly timed pause.'
nlp = spacy.load('en_core_web_md')

tagged_quote = [
    (int(t.pos_ == 'ADV'), t.pos_, t.text)  # <1>
    for t in nlp(quote)]

df_quote = pd.DataFrame(
    tagged_quote,
    columns='is_adv part_of_speech token'.split())

pd.options.display.max_rows = 20
print(df_quote)
nlp = spacy.load('en_core_web_md')

tagged_quote = [
    (int(t.pos_ == 'ADV'), t.pos_, t.text)  # <1>
    for t in nlp(quote)]

df_quote = pd.DataFrame(
    tagged_quote,
    columns='is_adv part_of_speech token'.split())

pd.options.display.max_rows = 20
print(df_quote.T)
nlp = spacy.load('en_core_web_md')

tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote.T)
nlp = spacy.load('en_core_web_md')

tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)
pd.options.display.max_columns = 8
nlp = spacy.load('en_core_web_md')

tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)
tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)
pd.options.display.max_columns = 10
tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)
hist
tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = [
    [t.text] + [p == t.pos_ for p in tags]
    for t in nlp(quote)]

df = pd.DataFrame(tagged_words,
    columns='is_adv is_adj is_noun token'.split()).T

pd.options.display.max_rows = 20
print(df_quote)
tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = [
    [t.text] + [p == t.pos_ for p in tags]
    for t in nlp(quote)]

df = pd.DataFrame(tagged_words,
    columns=).T

pd.options.display.max_rows = 20
print(df_quote)
tagged_words = {
    t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
    for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)
hist
tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = {
    tok.text: [int(tok.pos_ == tag) for tag in tags]  # <1>
    for tok in nlp(quote)}                            # <2>

df_quote = pd.DataFrame(tagged_words, index=tags)
print(df_quote)
conv = nn.Conv1d(in_channels=4, kernel_size=3, bias=False)
from torch import nn
conv = nn.Conv1d(in_channels=4, kernel_size=3, bias=False)
conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, bias=False)
print(conv.weight.size())
state = conv.state_dict()
state['weight'] = torch.tensor(kernel)
import torch
kernel = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
state['weight'] = torch.tensor(kernel).unsqueeze(0)
state['weight'].shape
kernel = [[1, 0, 0],
          [0, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
state['weight'] = torch.tensor(kernel).unsqueeze(0)
state['weight'].shape
cnn.load_state_dict(state)
conv.load_state_dict(state)
x
x = torch.tensor(df_quote)
x = torch.tensor(df_quote.values)
x = torch.tensor(df_quote.values.astype(float))
conv.forward(x)
x
conv.weight
conv.weight.dtype
state['weight'] = torch.tensor(kernel, dtype=torch.float64).unsqueeze(0)
conv.forward(x)
conv.load_state_dict(state)
conv.forward(x)
conv.weight
conv.weight.dtype
kernel = [[1, 0, 0.],
          [0, 0, 0.],
          [0, 1, 0.],
          [0, 0, 1.]]
state['weight'] = torch.tensor(kernel, dtype=torch.float64).unsqueeze(0)
conv.load_state_dict(state)
conv.weight.dtype
x.dtype
x.as_(torch.float32)
x.to(torch.float32)
x = x.to(torch.float32)
state['weight'] = torch.tensor(kernel).unsqueeze(0)
conv.load_state_dict(state)
conv.weight.dtype
conv.forward(x)
conv.forward(x).numpy()
np.array(conv.forward(x))
df
df = df_quote
df
df.loc['match'] = dict(zip(df.columns[:-2], conv.forward(x)))
y = np.array(conv.forward(x))
import numpy as np
y = np.array(conv.forward(x))
y = np.array(conv.forward(x).detach())
y
df.loc['match'] = dict(zip(df.columns[:-2], y[0]))
df
hist
hist -o -p -f ch07/ch07-conv1d-pos.hist.md
hist -f ch07/ch07-conv1d-pos.hist.py
