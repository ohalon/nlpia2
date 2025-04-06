import pandas as pd
import spacy
import numpy as np
nlp = spacy.load('en_core_web_md')  # <1>

text = 'right ones in the right order you can nudge the world'
doc = nlp(text)
df = pd.DataFrame([
    {k: getattr(t, k) for k in 'text pos_'.split()}
    for t in doc])

pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')

# .Python implementation of correlation
def corr(a, b):
    """ Compute the Pierson correlation coefficient R """
    a = a - np.mean(a)
    b = b - np.mean(b)
    return sum(a * b) / np.sqrt(sum(a*a) * sum(b*b))

a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

print(corr(a, b))
# 0.31622776601683794

print(corr(a, a))

nlp = spacy.load('en_core_web_md')

quote = "The right word may be effective, but no word was ever" \
 " as effective as a rightly timed pause."

tagged_words = {
   t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
   for t in nlp(quote)}                    # <2>

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)

#      The right  word  may   be     a rightly timed pause      .
# POS  DET   ADJ  NOUN  AUX  AUX   DET     ADV  VERB  NOUN  PUNCT
# ADV    0     0     0    0    0     0       1     0     0      0

# listing 7.3
inpt = list(df_quote.loc['ADV'])
print(inpt)

kernel = [.5, .5]                        # <1>

output = []
for i in range(len(inpt) - 1):           # <2>
   z = 0
   for k, weight in enumerate(kernel):  # <3>
       z = z + weight * inpt[i + k]
   output.append(z)

print(f'inpt:\n{inpt}')
print(f'len(inpt): {len(inpt)}')
print(f'output:\n{[int(o) if int(o)==o else o for o in output]}')
print(f'len(output): {len(output)}')

# inpt:
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 1, 1., 0, 0, 0., 1., 0, 0, 0]
# len(inpt): 20
# output:
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .5, 1, .5, 0, 0, .5, .5, 0, 0]
# len(output): 19

#Listing 7.5
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 120 # <1>

import seaborn as sns
sns.set_theme('paper')  # <2>

df = pd.DataFrame([inpt, output], index=['inpt', 'output']).T
ax = df.plot(style=['+-', 'o:'], linewidth=3)

# Listing 7.6
def convolve(inpt, kernel):
   output = []
   for i in range(len(inpt) - len(kernel) + 1):  # <1>
       output.append(
           sum(
               [
                   inpt[i + k] * kernel[k]
                   for k in range(len(kernel))   # <2>
               ]
           )
       )
   return output

tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = [
    [tok.text] + [int(tok.pos_ == tag) for tag in tags]  # <1>
    for tok in nlp(quote)]                                      # <2>

df = pd.DataFrame(tagged_words, columns=['token'] + tags).T
print(df)

#       The  right  word  may  be   a  rightly  timed  pause  .
# ADV     0      0     0    0   0   0        1      0      0  0
# ADJ     0      1     0    0   0   0        0      0      0  0
# VERB    0      0     0    0   0   0        0      1      0  0
# NOUN    0      0     1    0   0   0        0      0      1  0

# .Convert a DataFrame to a tensor with the correct size
import torch
x = x = torch.tensor(df.iloc[1:].astype(float).values, dtype=torch.float32)
x = x.unsqueeze(0)

kernel = pd.DataFrame(
           [[1, 0, 0.],
            [0, 0, 0.],
            [0, 1, 0.],
            [0, 0, 1.]], index=tags)
print(kernel)

# .Load hard-coded weights into a Conv1d layer
kernel = torch.tensor(kernel.values, dtype=torch.float32)
kernel = kernel.unsqueeze(0)   # <1>

conv = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, bias=False)
conv.load_state_dict({'weight': kernel})
print(conv.weight)

# tensor([[[1., 0., 0.],
#          [0., 0., 0.],
#          [0., 1., 0.],
#          [0., 0., 1.]]])

y = np.array(conv.forward(x).detach()).squeeze()
df.loc['y'] = pd.Series(y)
df

from nlpia2.init import maybe_download
url = 'https://upload.wikimedia.org/wikipedia/' \
      'commons/7/78/1210secretmorzecode.wav'
filepath = maybe_download(url, '1210secretmorzecode.wav')  # <1>
print(filepath)

from scipy.io import wavfile

sample_rate, audio = wavfile.read(filepath)
print(f'sample_rate: {sample_rate}')
print(f'audio:\n{audio}')

pd.options.display.max_rows = 7

audio = audio[:sample_rate * 2]                 # <1>
audio = np.abs(audio - audio.max() / 2) - .5    # <2>
audio = audio / audio.max()                     # <3>
audio = audio[::sample_rate // 400]             # <4>
audio = pd.Series(audio, name='audio')
audio.index = 1000 * audio.index / sample_rate  # <5>
audio.index.name = 'time (ms)'
print(f'sample_rate: {sample_rate}')
print(f'audio:\n{audio}')

kernel = [-1] * 24 + [1] * 24 + [-1] * 24                      # <1>
kernel = pd.Series(kernel, index=2.5 * np.arange(len(kernel)))
kernel.index.name = 'Time (ms)'
ax = kernel.plot(linewidth=3, ylabel='Kernel weight')

kernel = np.array(kernel) / sum(np.abs(kernel))  # <1>
pad = [0] * (len(kernel) // 2)                   # <2>
isdot = convolve(audio.values, kernel)
isdot =  np.array(pad[:-1] + list(isdot) + pad)  # <3>

df = pd.DataFrame()
df['audio'] = audio
df['isdot'] = isdot - isdot.min()
ax = df.plot()

isdot = np.convolve(audio.values, kernel, mode='same')  # <1>
df['isdot'] = isdot - isdot.min()
ax = df.plot()


df = pd.read_csv(HOME_DATA_DIR / 'news.csv')
df = df[['text', 'target']]  # <1>
print(df)

# Listing 7.17
import re
from collections import Counter
from itertools import chain
HOME_DATA_DIR = Path.home() / '.nlpia2-data'

counts = Counter(chain(*[
    re.findall(r'\w+', t.lower()) for t in df['text']]))     # <1>
vocab = [tok for tok, count in counts.most_common(4000)[3:]] # <2>

print(counts.most_common(10))

def pad(sequence, pad_value, seq_len):
    padded = list(sequence)[:seq_len]
    padded = padded + [pad_value] * (seq_len - len(padded))
    return padded

from torch import nn

embedding = nn.Embedding(
    num_embeddings=2000,    # <1>
    embedding_dim=64,       # <2>
    padding_idx=0)

from nlpia2.ch07.cnn.train79 import Pipeline  # <1>

pipeline = Pipeline(
    vocab_size=2000,
    embeddings=(2000, 64),
    epochs=7,
    torch_random_state=433994,  # <2>
    split_random_state=1460940,
)

pipeline = pipeline.train()

pipeline.epochs = 13  # <1>
pipeline = pipeline.train()

pipeline.indexes_to_texts(pipeline.x_test[:4])

def describe_model(model):  # <1>
    state = model.state_dict()
    names = state.keys()
    weights = state.values()
    params = model.parameters()
    df = pd.DataFrame([
        dict(
            name=name,
            learned_params=int(p.requires_grad) * p.numel(),  # <2>
            all_params=p.numel(),  # <3>
            size=p.size(),
        )
        for name, w, p in zip(names, weights, params)
    ]
    )
    df = df.set_index('name')
    return df

describe_model(pipeline.model)  # <4>

from nessvec.files import load_vecs_df
glove = load_vecs_df(HOME_DATA_DIR / 'glove.6B.50d.txt')
zeroes = [0.] * 50
embed = []
for tok in vocab:                     # <1>
    if tok in glove.index:
        embed.append(glove.loc[tok])
    else:
        embed.append(zeros)           # <2>
embed = np.array(embed)

print(f'embed.shape: {embed.shape}')
print(f'vocab:\n{pd.Series(vocab)}')

embed = torch.Tensor(embed)                         # <1>
print(f'embed.size(): {embed.size()}')
embed = nn.Embedding.from_pretrained(embed, freeze=False)  # <2>
print(embed)

class CNNTextClassifier(nn.Module):

    def __init__(self, embeddings):
        super().__init__()

        self.seq_len = 40                               # <1>
        self.vocab_size = 10000                         # <2>
        self.embedding_size = 50                        # <3>
        self.out_channels = 5                           # <4>
        self.kernel_lengths = [2, 3, 4, 5, 6]           # <5>
        self.stride = 1                                 # <6>
        self.dropout = nn.Dropout(0)                    # <7>
        self.pool_stride = self.stride                  # <8>
        self.conv_out_seq_len = calc_out_seq_len(       # <9>
            seq_len=self.seq_len,
            kernel_lengths=self.kernel_lengths,
            stride=self.stride,
            )
        self.embed = nn.Embedding(
            self.vocab_size,  # <1>
            self.embedding_size,  # <2>
            padding_idx=0)
        state = self.embed.state_dict()
        state['weight'] = embeddings  # <3>
        self.embed.load_state_dict(state)
        self.convolvers = []
        self.poolers = []
        total_out_len = 0
        for i, kernel_len in enumerate(self.kernel_lengths):
            self.convolvers.append(
                nn.Conv1d(in_channels=self.embedding_size,
                          out_channels=self.out_channels,
                          kernel_size=kernel_len,
                          stride=self.stride))
            print(f'conv[{i}].weight.shape: {self.convolvers[-1].weight.shape}')
            conv_output_len = calc_conv_out_seq_len(
                seq_len=self.seq_len, kernel_len=kernel_len, stride=self.stride)
            print(f'conv_output_len: {conv_output_len}')
            self.poolers.append(
                nn.MaxPool1d(kernel_size=conv_output_len, stride=self.stride))
            total_out_len += calc_conv_out_seq_len(
                seq_len=conv_output_len, kernel_len=conv_output_len,
                stride=self.stride)
            print(f'total_out_len: {total_out_len}')
            print(f'poolers[{i}]: {self.poolers[-1]}')
        print(f'total_out_len: {total_out_len}')
        self.linear_layer = nn.Linear(self.out_channels * total_out_len, 1)
        print(f'linear_layer: {self.linear_layer}')

def calc_conv_out_seq_len(seq_len, kernel_len,
                          stride=1, dilation=1, padding=0):
    """
    L_out =     (L_in + 2 * padding - dilation * (kernel_size - 1) - 1)
            1 + _______________________________________________________
                                        stride
    """
    return (
        1 + (seq_len +
             2 * padding - dilation * (kernel_len - 1) - 1
            ) //
        stride
        )

