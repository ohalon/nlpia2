# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt


# num_examples = 7
num_channels = 1
# embedding_size = 1

kernel = [-2, 3]
kernel_size = len(kernel)
stride = 1

sentence_len = 11
pad_len = 4
seq_len = sentence_len + pad_len

x = np.array(list(range(sentence_len)) + [0] * pad_len, np.float32)
x = torch.tensor(x)
print()
print(f"x.resize_({num_channels}, {num_channels}, {seq_len})")
print(x.resize_(num_channels, num_channels, seq_len))
print()
print('x')
print(x)

conv = nn.Conv1d(
    in_channels=num_channels,
    out_channels=num_channels,
    # groups=None,
    stride=1,
    kernel_size=2
)
print()
print(f"conv = nn.Conv1d({num_channels}, {num_channels}, stride={stride}, kernel_size={kernel_size})")
print(conv)

print()
print('conv.weight')
print(conv.weight)
print()
print('conv.bias')
print(conv.bias)

print()
print('conv.forward(x)')
print(conv.forward(x))

state = conv.state_dict()
print()
print('state (conv.state_dict()):')
print(state)

state['weight'] = torch.tensor(np.array([[kernel]], dtype=np.float32))
state['bias'] = torch.tensor([0])
print()
print('updated state:')
print(state)

conv.load_state_dict(state)
print()
print('updated conv:')
print(conv)

x = conv.forward(x)
print('x = conv.forward(x): ')
print(x)

pool_size = 3
pool_stride = 2
pool = nn.MaxPool1d(pool_size, pool_stride)
print(f"pool = nn.MaxPool1d({pool_size}, {pool_stride})")
print(pool)

y = pool.forward(x)
print('y = pool.forward(x): ')
print(y)
