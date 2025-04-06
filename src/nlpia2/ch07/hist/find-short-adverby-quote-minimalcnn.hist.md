>>> # !pip install dataframe-image
... import dataframe_image as dfi
... 
... import pandas as pd
... import numpy as np
... import torch
... from torch import nn
... from pathlib import Path
... from matplotlib import pyplot as plt
... 
... 
... num_examples = 7
... seq_len = 5
... embedding_size = 1
... 
... dataset = torch.arange(
...     num_examples * seq_len * embedding_size,
...     dtype=torch.float)
... dataset.resize_(num_examples, seq_len, embedding_size)
... 
... df = pd.DataFrame(np.arange(
...     num_examples * seq_len * embedding_size,
...     dtype=float).reshape(num_examples, seq_len * embedding_size))
... IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'
... 
... dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
... dataset = torch.from_numpy(df.values).resize(num_examples, seq_len, embedding_size)
... 
... x = dataset[0]
... x.resize_(seq_len, embedding_size)
... 
... conv = nn.Conv1d(in_channels=embedding_size, out_channels=1, groups=None, stride=1, kernel_size=2)
... print(conv.weight)
...
>>> hist
>>> num_examples = 7
... seq_len = 5
... embedding_size = 1
... 
... dataset = torch.arange(
...     num_examples * seq_len * embedding_size,
...     dtype=torch.float)
... dataset.resize_(num_examples, seq_len, embedding_size)
... 
... df = pd.DataFrame(np.arange(
...     num_examples * seq_len * embedding_size,
...     dtype=float).reshape(num_examples, seq_len * embedding_size))
... IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'
... 
... dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
... dataset = torch.from_numpy(df.values).resize(num_examples, seq_len, embedding_size)
... 
... x = dataset[0]
... x.resize_(seq_len, embedding_size)
...
tensor([[0.],
        [1.],
        [2.],
        [3.],
        [4.]], dtype=torch.float64)
>>> conv = nn.Conv1d(
...     in_channels=embedding_size,
...     out_channels=1,
...     # groups=None,
...     stride=1,
...     kernel_size=2
... )
... print(conv.weight)
...
>>> conv.state_dict()
OrderedDict([('weight', tensor([[[-0.1631, -0.1114]]])),
             ('bias', tensor([0.1136]))])
>>> state = conv.state_dict()
>>> state['bias'] = torch.tensor([0.1])
>>> conv.load_state_dict(state)
<All keys matched successfully>
>>> state['weight'] = torch.tensor([[[1., 2.]]])
>>> state['weight'] = torch.tensor([[[-1., -2.]]])
>>> x
tensor([[0.],
        [1.],
        [2.],
        [3.],
        [4.]], dtype=torch.float64)
>>> conv.forward(x)
>>> x.size()
torch.Size([5, 1])
>>> x.resize_([1, 5, 1])
tensor([[[0.],
         [1.],
         [2.],
         [3.],
         [4.]]], dtype=torch.float64)
>>> conv.forward(x)
>>> x.size()
torch.Size([1, 5, 1])
>>> x.resize_([1, 1, 5])
tensor([[[0., 1., 2., 3., 4.]]], dtype=torch.float64)
>>> x.size()
torch.Size([1, 1, 5])
>>> conv.forward(x)
>>> conv = nn.Conv1d(
...     in_channels=embedding_size,
...     out_channels=1,
...     # groups=None,
...     stride=1,
...     kernel_size=2
... )
... print(conv.weight)
...
>>> conv.weight.dtype
torch.float32
>>> conv.bias.dtype
torch.float32
>>> x.dtype
torch.float64
>>> data = np.arange(
...     num_examples * seq_len * embedding_size,
...     dtype=np.float32,
... )
... data = data.reshape(num_examples, seq_len * embedding_size)
... df = pd.DataFrame(data)
... 
... IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'
... dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
... 
... dataset = torch.from_numpy(df.values)
... dataset.resize_(num_examples, seq_len, embedding_size)
... 
... x = dataset[0]
... x.resize_(seq_len, embedding_size)
... 
... conv = nn.Conv1d(
...     in_channels=embedding_size,
...     out_channels=1,
...     # groups=None,
...     stride=1,
...     kernel_size=2
... )
... print(conv.weight)
... 
... conv.forward(x)
...
>>> x.resize_(num_channels, embedding_size, seq_len)
>>> num_channels = 1
>>> x.resize_(num_channels, embedding_size, seq_len)
tensor([[[0., 1., 2., 3., 4.]]])
>>> conv.forward(x)
tensor([[[0.7927, 2.0346, 3.2766, 4.5185]]], grad_fn=<ConvolutionBackward0>)
>>> state = conv.state_dict()
>>> state['weight'] = torch.tensor(np.array([[[-1., -2.]]], dtype=np.float32))
>>> conv.load_state_dict(state)
<All keys matched successfully>
>>> conv.forward(x)
tensor([[[ -1.8770,  -4.8770,  -7.8770, -10.8770]]],
       grad_fn=<ConvolutionBackward0>)
>>> state['bias'] = torch.tensor([0.1])
>>> conv.forward(x)
tensor([[[ -1.8770,  -4.8770,  -7.8770, -10.8770]]],
       grad_fn=<ConvolutionBackward0>)
>>> conv.state_dict()
OrderedDict([('weight', tensor([[[-1., -2.]]])), ('bias', tensor([0.1230]))])
>>> state['bias'] = torch.tensor([0.1])
>>> conv.load_state_dict(state)
<All keys matched successfully>
>>> conv.forward(x)
tensor([[[ -1.9000,  -4.9000,  -7.9000, -10.9000]]],
       grad_fn=<ConvolutionBackward0>)
>>> state['weight'] = torch.tensor(np.array([[[-2, 3]]], dtype=np.float32))
>>> state['bias'] = torch.tensor([0])
>>> conv.load_state_dict(state)
<All keys matched successfully>
>>> conv.forward(x)
tensor([[[3., 4., 5., 6.]]], grad_fn=<ConvolutionBackward0>)
>>> x
tensor([[[0., 1., 2., 3., 4.]]])
>>> hist
>>> %run minimalcnn
>>> x
tensor([[[0., 1., 2., 3., 4.]]])
>>> %run minimalcnn
>>> %run minimalcnn
>>> %run minimalcnn
>>> nn.MaxPool1d(kernel_len, stride)
>>> nn.MaxPool1d(kernel_size, stride)
MaxPool1d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
>>> pool = nn.MaxPool1d(kernel_size, stride)
>>> pool.forward(y)
tensor([[[0.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000]]],
       grad_fn=<SqueezeBackward1>)
>>> pool = nn.MaxPool1d(pool_size, pool_stride)
>>> pool_size = 3
>>> pool_stride = 2
>>> pool = nn.MaxPool1d(pool_size, pool_stride)
>>> pool.forward(y)
tensor([[[0.5000, 0.5000, 0.0000]]], grad_fn=<SqueezeBackward1>)
>>> hist
>>> import spacy
>>> nlp = spacy.load('en_core_web_md')
>>> spacy.cli.download('en_core_web_md')
>>> nlp = spacy.load('en_core_web_md')
>>> from nlpia2.init import maybe_download, DATA_DIR, HOME_DATA_DIR
>>> maybe_download('quotes.yml')
PosixPath('/home/hobs/.nlpia2-data/quotes.yml')
>>> quotes = maybe_download('quotes.yml')
>>> for q in quotes:
...     print(q['text'])
...
>>> quotes = yaml.full_load(maybe_download('quotes.yml').open())
>>> import yaml
>>> quotes = yaml.full_load(maybe_download('quotes.yml').open())
>>> for q in quotes:
...     print(q['text'])
...
>>> for q in quotes:
...     print([t.pos_ for t in q['text']])
...
>>> for q in quotes:
...     print([t.pos_ for t in nlp(q['text'])])
...
>>> df = []
... for q in quotes:
...     df.append([t.pos_ for t in nlp(q['text'])])
...
>>> df[0:2]
[['PRON',
  'AUX',
  'DET',
  'NOUN',
  'PUNCT',
  'PROPN',
  'PROPN',
  'PUNCT',
  'PRON',
  'AUX',
  'NOUN',
  'DET',
  'DET',
  'NOUN',
  'ADV',
  'PUNCT'],
 ['SCONJ',
  'PRON',
  'AUX',
  'VERB',
  'DET',
  'NOUN',
  'PUNCT',
  'ADV',
  'PROPN',
  'AUX',
  'VERB',
  'DET',
  'NOUN',
  'PUNCT',
  'SCONJ',
  'PRON',
  'AUX',
  'AUX',
  'PRON',
  'ADP',
  'DET',
  'NOUN',
  'PUNCT',
  'PRON',
  'AUX',
  'ADV',
  'ADV',
  'ADV',
  'AUX',
  'DET',
  'NOUN',
  'ADP',
  'PROPN',
  'PUNCT',
  'SCONJ',
  'SCONJ',
  'PRON',
  'AUX',
  'PART',
  'AUX',
  'DET',
  'NOUN',
  'ADP',
  'DET',
  'NOUN',
  'PUNCT']]
>>> df = []
... for q in quotes:
...     df.append([1 if t.pos_ == 'ADJ' else 0 for t in nlp(q['text'])])
...
>>> df = []
... for q in quotes:
...     text = q['text']
...     df.append([t.pos_ for t in nlp(text)])
...     df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
...
>>> df = pd.DataFrame(df)
>>> df
     0    1    2    3    4    5    6    7    8    9    ...  437  438  439  440  441  442  443  444  445  446
0      0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
1      0    0    0    0  0.0  0.0  0.0  1.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
2      0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
3      0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
4      0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
..   ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
239    0    0    1    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
240    0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
241    0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
242    0    0    0    0  0.0  0.0  0.0  0.0  0.0  1.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
243    0    0    0    0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN

[244 rows x 447 columns]
>>> df.T.sum()
0      1.0
1      4.0
2      0.0
3      0.0
4      1.0
      ... 
239    3.0
240    0.0
241    0.0
242    7.0
243    3.0
Length: 244, dtype: float64
>>> df.T.sum() / df.T.apply(len)
0      0.002237
1      0.008949
2      0.000000
3      0.000000
4      0.002237
         ...   
239    0.006711
240    0.000000
241    0.000000
242    0.015660
243    0.006711
Length: 244, dtype: float64
>>> df = []
... adverby_quotes = []
... for i, q in enumerate(quotes):
...     text = q['text']
...     df.append([t.pos_ for t in nlp(text)])
...     df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
...     bits = df[-1]
...     if (1, 1) in zip(bits[:-1], bits[1:]):
...         adverby_quotes.append(i)
... df.iloc[adverby_quotes]
...
>>> df = []
... adverby_quotes = []
... for i, q in enumerate(quotes):
...     text = q['text']
...     df.append([t.pos_ for t in nlp(text)])
...     df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
...     bits = df[-1]
...     if (1, 1) in zip(bits[:-1], bits[1:]):
...         adverby_quotes.append(i)
... np.array(df)[adverby_quotes]
...
array([list([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 1, 1, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]),
       list([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
       list([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]),
       list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
       list([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
       list([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
       list([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
       list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
       list([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
       list([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])],
      dtype=object)
>>> df = []
... adverby_quotes = []
... for i, q in enumerate(quotes):
...     text = q['text']
...     df.append([t.pos_ for t in nlp(text)])
...     df[-1] = [1 if p == 'ADV' else 0 for p in df[-1]]
...     bits = df[-1]
...     if (1, 1) in zip(bits[:-1], bits[1:]):
...         adverby_quotes.append(i)
...
>>> df = pd.DataFrame(df).iloc[adverby_quotes]
>>> (df.iloc[0] > -1).sum()
46
>>> (df.iloc[1] > -1).sum()
57
>>> df['len'] = (df > -1).sum(axis=1)
>>> df.len
1       46
22      57
24     133
41       9
65      21
67      18
81      30
86      58
89     112
92      20
93      37
134     29
140     29
156     29
162     29
171     77
181     75
189     39
190     90
212    447
215     88
222     25
223     30
228     45
231     24
233     43
242    150
Name: len, dtype: int64
>>> df.sort_values('len')
     0  1  2  3    4    5    6    7    8    9   10  ...  437  438  439  440  441  442  443  444  445  446  len
41   0  1  1  0  0.0  0.0  0.0  0.0  0.0  NaN  NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN    9
67   0  0  1  1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   18
92   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   20
65   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   21
231  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   24
222  0  0  0  0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   25
156  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
134  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
140  1  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
162  1  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
81   0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   30
223  1  1  0  1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   30
93   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   37
189  0  0  0  0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   39
233  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   43
228  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   45
1    0  0  0  0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   46
22   0  0  0  1  0.0  1.0  1.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   57
86   0  0  0  0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   58
181  0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   75
171  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   77
215  0  0  0  0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   88
190  0  0  1  0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   90
89   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  112
24   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  133
242  0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  150
212  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  447

[27 rows x 448 columns]
>>> len(df)
27
>>> df.sort_values('len')[:5]
     0  1  2  3    4    5    6    7    8    9   10  ...  437  438  439  440  441  442  443  444  445  446  len
41   0  1  1  0  0.0  0.0  0.0  0.0  0.0  NaN  NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN    9
67   0  0  1  1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   18
92   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   20
65   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   21
231  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   24

[5 rows x 448 columns]
>>> np.array(quotes)[df.sort_values('len')[:5].index.values]
array([{'text': 'Ruptures almost always lead to a stronger project.', 'author': 'Anne Carson', 'source': 'Every Day a Word Surprises Me & Other Quotes by Writers', 'tags': ['project management', 'agile', 'fail fast', 'failure']},
       {'text': 'We are more often frightened than hurt; and we suffer more in imagination than in reality.', 'author': 'Seneca', 'tags': ['stoicism', 'inspiration', 'stoic', 'philosophy', 'life', 'mindfulness']},
       {'text': 'The right word may be effective, but no word was ever as effective as a rightly timed pause.', 'author': 'Mark Twain', 'tags': ['nlp', 'nlpia', 'ch1', 'ch6', 'ch7', 'stop-words', 'word sequence', 'n-gram', 'rnn', 'words', 'power of words', 'power of pen', 'communication', 'thought']},
       {'text': "External things are not the problem. It's your assessment of them. Which you can erase right now.", 'author': 'Marcus Aurelius', 'tags': ['stoicism', 'inspiration', 'stoic', 'philosophy', 'life', 'mindfulness']},
       {'text': 'As long as you give it enough time, life is stronger than metal and stone, more powerful than typhoons and volcanoes.', 'tags': ['life', 'survival of the fittest', 'survival', 'death', 'meaning', 'scale', 'intelligence', 'big picture', 'power', 'evolution', 'time', 'universe', 'optimism'], 'page': 31, 'book': "Death's End", 'character': 'Yang Dong', 'author': 'Cixin Liu'}],
      dtype=object)
>>> hist -o -p -f find-short-adverby-quote.yml
>>> hist -o -p -f find-short-adverby-quote.hist.md
>>> hist -f find-short-adverby-quote.hist.py
>>> hist
>>> ls -hal
>>> rm find-short-adverby-quote.yml
>>> more find-short-adverby-quote.hist.py
>>> more find-short-adverby-quote.hist.md
>>> df
     0  1  2  3    4    5    6    7    8    9   10  ...  437  438  439  440  441  442  443  444  445  446  len
1    0  0  0  0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   46
22   0  0  0  1  0.0  1.0  1.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   57
24   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  133
41   0  1  1  0  0.0  0.0  0.0  0.0  0.0  NaN  NaN  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN    9
65   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   21
67   0  0  1  1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   18
81   0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   30
86   0  0  0  0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   58
89   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  112
92   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   20
93   0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   37
134  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
140  1  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
156  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
162  1  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   29
171  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   77
181  0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  1.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   75
189  0  0  0  0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   39
190  0  0  1  0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   90
212  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  447
215  0  0  0  0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   88
222  0  0  0  0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   25
223  1  1  0  1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   30
228  0  0  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   45
231  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   24
233  1  1  0  0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   43
242  0  0  0  0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  ...  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  150

[27 rows x 448 columns]
>>> df.index[3]
41
>>> pd.options.display.max_rows = 7
>>> pd.options.display.max_cols = 7
>>> pd.options.display.max_columns = 7
>>> df
     0  1  2  ...  445  446  len
1    0  0  0  ...  NaN  NaN   46
22   0  0  0  ...  NaN  NaN   57
24   0  0  0  ...  NaN  NaN  133
..  .. .. ..  ...  ...  ...  ...
231  1  1  0  ...  NaN  NaN   24
233  1  1  0  ...  NaN  NaN   43
242  0  0  0  ...  NaN  NaN  150

[27 rows x 448 columns]
>>> df = df.sort_values('len')
>>> shortest5 = df[:5].index.values]
>>> shortest5 = df[:5].index.values
>>> shortest5
array([ 41,  67,  92,  65, 231])
>>> quotes[92]
{'text': 'The right word may be effective, but no word was ever as effective as a rightly timed pause.',
 'author': 'Mark Twain',
 'tags': ['nlp',
  'nlpia',
  'ch1',
  'ch6',
  'ch7',
  'stop-words',
  'word sequence',
  'n-gram',
  'rnn',
  'words',
  'power of words',
  'power of pen',
  'communication',
  'thought']}
>>> hist -o -p -f 'find-short-adverby-quote-and-minimalcnn.hist.md'
>>> hist -f 'find-short-adverby-quote-and-minimalcnn.hist.py'
>>> more 'find-short-adverby-quote-and-minimalcnn.hist.py'
>>> ls find*
>>> hist -o -p -f find-short-adverby-quote-minimalcnn.hist.md
>>> hist -f find-short-adverby-quote-minimalcnn.hist.py
>>> ls
>>> ls find*
>>> rm find-short-adverby-quote.hist.md
>>> rm find-short-adverby-quote.hist.py
>>> more find-short-adverby-quote-minimalcnn.hist.md
>>> quotes[92]
{'text': 'The right word may be effective, but no word was ever as effective as a rightly timed pause.',
 'author': 'Mark Twain',
 'tags': ['nlp',
  'nlpia',
  'ch1',
  'ch6',
  'ch7',
  'stop-words',
  'word sequence',
  'n-gram',
  'rnn',
  'words',
  'power of words',
  'power of pen',
  'communication',
  'thought']}
>>> df.iloc[92]
>>> df.loc[92]
0       0.0
1       0.0
2       0.0
       ... 
445     NaN
446     NaN
len    20.0
Name: 92, Length: 448, dtype: float64
>>> df.loc[92].dropna()
0       0.0
1       0.0
2       0.0
       ... 
18      0.0
19      0.0
len    20.0
Name: 92, Length: 21, dtype: float64
>>> df.loc[92].dropna()[-5:]
16      1.0
17      0.0
18      0.0
19      0.0
len    20.0
Name: 92, dtype: float64
>>> df.loc[92].dropna()[-7:]
14      0.0
15      0.0
16      1.0
17      0.0
18      0.0
19      0.0
len    20.0
Name: 92, dtype: float64
>>> list(df.loc[92].dropna())
[0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 1.0,
 1.0,
 0.0,
 0.0,
 0.0,
 1.0,
 0.0,
 0.0,
 0.0,
 20.0]
>>> quote = quotes[92]
>>> doc = nlp(quote['text'])
>>> quote['tags']
['nlp',
 'nlpia',
 'ch1',
 'ch6',
 'ch7',
 'stop-words',
 'word sequence',
 'n-gram',
 'rnn',
 'words',
 'power of words',
 'power of pen',
 'communication',
 'thought']
>>> [(t.text, t.pos_, t.pos_ == 'ADV') for t in doc]
[('The', 'DET', False),
 ('right', 'ADJ', False),
 ('word', 'NOUN', False),
 ('may', 'AUX', False),
 ('be', 'AUX', False),
 ('effective', 'ADJ', False),
 (',', 'PUNCT', False),
 ('but', 'CCONJ', False),
 ('no', 'DET', False),
 ('word', 'NOUN', False),
 ('was', 'AUX', False),
 ('ever', 'ADV', True),
 ('as', 'ADV', True),
 ('effective', 'ADJ', False),
 ('as', 'ADP', False),
 ('a', 'DET', False),
 ('rightly', 'ADV', True),
 ('timed', 'VERB', False),
 ('pause', 'NOUN', False),
 ('.', 'PUNCT', False)]
>>> tagged_quote = [(t.text, t.pos_, int(t.pos_ == 'ADV')) for t in doc]
>>> tagged_quote
[('The', 'DET', 0),
 ('right', 'ADJ', 0),
 ('word', 'NOUN', 0),
 ('may', 'AUX', 0),
 ('be', 'AUX', 0),
 ('effective', 'ADJ', 0),
 (',', 'PUNCT', 0),
 ('but', 'CCONJ', 0),
 ('no', 'DET', 0),
 ('word', 'NOUN', 0),
 ('was', 'AUX', 0),
 ('ever', 'ADV', 1),
 ('as', 'ADV', 1),
 ('effective', 'ADJ', 0),
 ('as', 'ADP', 0),
 ('a', 'DET', 0),
 ('rightly', 'ADV', 1),
 ('timed', 'VERB', 0),
 ('pause', 'NOUN', 0),
 ('.', 'PUNCT', 0)]
>>> tagged_quote = [(int(t.pos_ == 'ADV'), t.pos_, t.text) for t in doc]
>>> tagged_quote
[(0, 'DET', 'The'),
 (0, 'ADJ', 'right'),
 (0, 'NOUN', 'word'),
 (0, 'AUX', 'may'),
 (0, 'AUX', 'be'),
 (0, 'ADJ', 'effective'),
 (0, 'PUNCT', ','),
 (0, 'CCONJ', 'but'),
 (0, 'DET', 'no'),
 (0, 'NOUN', 'word'),
 (0, 'AUX', 'was'),
 (1, 'ADV', 'ever'),
 (1, 'ADV', 'as'),
 (0, 'ADJ', 'effective'),
 (0, 'ADP', 'as'),
 (0, 'DET', 'a'),
 (1, 'ADV', 'rightly'),
 (0, 'VERB', 'timed'),
 (0, 'NOUN', 'pause'),
 (0, 'PUNCT', '.')]
>>> pd.DataFrame(tagged_quote)
    0      1      2
0   0    DET    The
1   0    ADJ  right
2   0   NOUN   word
.. ..    ...    ...
17  0   VERB  timed
18  0   NOUN  pause
19  0  PUNCT      .

[20 rows x 3 columns]
>>> pd.options.display.max_rows = 21
>>> pd.DataFrame(tagged_quote)
    0      1          2
0   0    DET        The
1   0    ADJ      right
2   0   NOUN       word
3   0    AUX        may
4   0    AUX         be
5   0    ADJ  effective
6   0  PUNCT          ,
7   0  CCONJ        but
8   0    DET         no
9   0   NOUN       word
10  0    AUX        was
11  1    ADV       ever
12  1    ADV         as
13  0    ADJ  effective
14  0    ADP         as
15  0    DET          a
16  1    ADV    rightly
17  0   VERB      timed
18  0   NOUN      pause
19  0  PUNCT          .
>>> pd.DataFrame(tagged_quote, columns='is_adv pos word'.split())
    is_adv    pos       word
0        0    DET        The
1        0    ADJ      right
2        0   NOUN       word
3        0    AUX        may
4        0    AUX         be
5        0    ADJ  effective
6        0  PUNCT          ,
7        0  CCONJ        but
8        0    DET         no
9        0   NOUN       word
10       0    AUX        was
11       1    ADV       ever
12       1    ADV         as
13       0    ADJ  effective
14       0    ADP         as
15       0    DET          a
16       1    ADV    rightly
17       0   VERB      timed
18       0   NOUN      pause
19       0  PUNCT          .
>>> pd.DataFrame(tagged_quote, columns='is_adv POS word'.split())
    is_adv    POS       word
0        0    DET        The
1        0    ADJ      right
2        0   NOUN       word
3        0    AUX        may
4        0    AUX         be
5        0    ADJ  effective
6        0  PUNCT          ,
7        0  CCONJ        but
8        0    DET         no
9        0   NOUN       word
10       0    AUX        was
11       1    ADV       ever
12       1    ADV         as
13       0    ADJ  effective
14       0    ADP         as
15       0    DET          a
16       1    ADV    rightly
17       0   VERB      timed
18       0   NOUN      pause
19       0  PUNCT          .
>>> hist -f find-short-adverby-quote-minimalcnn.hist.py
>>> hist -o -p -f find-short-adverby-quote-minimalcnn.hist.md
