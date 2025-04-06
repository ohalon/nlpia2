# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt  # noqa

IMAGES_DIR = Path.home() / 'code' / 'tangibleai' / 'nlpia-manuscript' / 'manuscript' / 'images' / 'ch07'

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

dfi.export(df, IMAGES_DIR / 'df-minimal-cnn-dataset.png', max_rows=7)
dataset = torch.from_numpy(df.values).resize(num_examples, seq_len, embedding_size)

kernel = [-2, 3]
x = [4, 6, 8, 10, 12, 14, 0, 0, 0]
seq_len = len(x)

kernel_size = len(kernel)
x = torch.tensor(x, dtype=torch.float32)


conv = nn.Conv1d(
    in_channels=embedding_size,
    out_channels=1,
    # groups=None,
    stride=1,
    kernel_size=kernel_size,
)
print(conv.weight)
state = conv.state_dict()
print(state)

# x.resize_([1, 1, 5])
# x = x.to(torch.float32)
# print(x)
# print(conv.weight)
# conv.forward(x)


num_channels = 1
x.resize_(num_channels, embedding_size, seq_len)

state['weight'] = torch.tensor(np.array([[kernel]], dtype=np.float32))
state['bias'] = torch.tensor(np.array([0], dtype=np.float32))
conv.load_state_dict(state)
x1 = conv.forward(x)
print(x1)
pool_size = 3
pool_stride = 2
pool = nn.MaxPool1d(pool_size, pool_stride)
y = pool.forward(x1)
