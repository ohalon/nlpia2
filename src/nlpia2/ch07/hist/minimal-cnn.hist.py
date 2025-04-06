# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt
hist
x = torch.range(0, 31)
x.resize(8, 4)
hist

x = torch.range(1, 35)
x.resize(7, 5)


class MinimalCNN(nn.Module):

    def __init__(self, vocab_size=5, embedding_size=50):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=50,
            padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=50,
            out_channels=50,
            groups=50,
            kernel_size=2,
            stride=1)

    def forward(self, x):
        embeddings = self.embedding(x)
        print(f"embeddings.size(): {embeddings.size()}")
        print(f"embeddings:\n{embeddings}")
        features = self.conv(embeddings)
        print(f"features.size(): {features.size()}")
        print(f"features:\n{features}")
        return features.squeeze()
#             z = torch.relu(z)
#             z = pool(z)
#             conv_outputs.append(z)
cnn = MinimalCNN()
cnn.forward(x)
%run minimalcnn
ls
cd ..
%run minimalcnn
%run minimalcnn
x
x.resize?
dir(x.resize)
dir(x)
help(x.resize)
x.resize_?
%run minimalcnn
%run minimalcnn
x
cnn = MinimalCNN()
cnn.forward(x)
%run minimalcnn
cnn = MinimalCNN()
cnn.forward(x)
%run minimalcnn
%run minimalcnn
cnn.conv.weight
cnn.conv.weight.size()
hist
cnn.conv.weight.size()
x
%run minimalcnn
cnn.conv.weight.size()
%run minimalcnn
cnn.conv.weight.set_(torch.ones_like(cnn.conv.weight))
torch.prod(cnn.conv.weight, torch.zeros_like(cnn.conv.weight))
torch.prod(cnn.conv.weight, torch.tensor(torch.zeros_like(cnn.conv.weight)))
torch.prod(torch.tensor(cnn.conv.weight), torch.tensor(torch.zeros_like(cnn.conv.weight)))
type(cnn.conv.weight)
cnn.conv.weight.to_dense()
cnn.conv.weight.to_dense
cnn.conv.weight.tensor_split()
cnn.conv.weight.tensor_split
cnn.conv.weight.as_strided()
cnn.conv.weight.as_strided(1)
cnn.conv.weight.as_strided((1,))
cnn.conv.weight.as_strided(stride=1)
help(cnn.conv.weight)
cnn.conv.weight.super()
cnn.conv.weight.tensor_split?
cnn.conv.weight.tensor_split(0, 0)
cnn.conv.weight.tensor_split(1, 0)
type(cnn.conv.weight.tensor_split(1, 0))
cnn.conv.weight.data = torch.arange(torch.cumprod(cnn.conv.weight.data.size()))
cnn.conv.weight.data = torch.arange(torch.cumprod(list(cnn.conv.weight.data.size())))
cnn.conv.weight.data = torch.arange(torch.cumprod(torch.tensor(list(cnn.conv.weight.data.size()))))
cnn.conv.weight.data = torch.arange(torch.cumprod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()])))
torch.tensor([.1 * w for w in cnn.conv.weight.data.size()])
cnn.conv.weight.data.size()
torch.cumprod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()]))
torch.prod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()]))
torch.prod(torch.tensor([w for w in cnn.conv.weight.data.size()]))
torch.prod(torch.tensor(cnn.conv.weight.data.size()))
torch.arange(torch.prod(torch.tensor(cnn.conv.weight.data.size())))
torch.arange(torch.prod(torch.tensor(cnn.conv.weight.data.size())))
w = _
w.to(float)
cnn.conv.weight.data.set_(w.to(float))
cnn.conv.weight.data.set_(w.to(float))
w = w.to(float)
w.resize_as(cnn.conv.weight.data)
cnn.conv.weight.data.set_(w.resize_as(cnn.conv.weight.data))
hist
pwd
hist -o -p -f minimal-cnn.hist.md
hist -f minimal-cnn.hist.py
