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
>>> hist
>>> x = torch.range(0, 31)
>>> x.resize(8, 4)
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.],
        [12., 13., 14., 15.],
        [16., 17., 18., 19.],
        [20., 21., 22., 23.],
        [24., 25., 26., 27.],
        [28., 29., 30., 31.]])
>>> hist
>>> 
... x = torch.range(1, 35)
... x.resize(7, 5)
... 
... 
... class MinimalCNN(nn.Module):
... 
...     def __init__(self, vocab_size=5, embedding_size=50):
...         super().__init__()
... 
...         self.embedding = nn.Embedding(
...             num_embeddings=5,
...             embedding_dim=50,
...             padding_idx=0)
...         self.conv = nn.Conv1d(
...             in_channels=50,
...             out_channels=50,
...             groups=50,
...             kernel_size=2,
...             stride=1)
... 
...     def forward(self, x):
...         embeddings = self.embedding(x)
...         print(f"embeddings.size(): {embeddings.size()}")
...         print(f"embeddings:\n{embeddings}")
...         features = self.conv(embeddings)
...         print(f"features.size(): {features.size()}")
...         print(f"features:\n{features}")
...         return features.squeeze()
... #             z = torch.relu(z)
... #             z = pool(z)
... #             conv_outputs.append(z)
...
>>> cnn = MinimalCNN()
>>> cnn.forward(x)
>>> %run minimalcnn
>>> ls
>>> cd ..
>>> %run minimalcnn
>>> %run minimalcnn
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
        15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28.,
        29., 30., 31., 32., 33., 34., 35.])
>>> x.resize?
>>> x
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34]])
>>> cnn = MinimalCNN()
>>> cnn.forward(x)
>>> %run minimalcnn
>>> cnn = MinimalCNN()
>>> cnn.forward(x)
>>> %run minimalcnn
>>> %run minimalcnn
>>> cnn.conv.weight
Parameter containing:
tensor([[[-0.0223,  0.2827],
         [-0.1783,  0.1068],
         [-0.0757,  0.1051],
         [-0.3135,  0.2356],
         [ 0.0809, -0.1135]],

        [[-0.2034, -0.0933],
         [-0.3091, -0.0916],
         [-0.1160, -0.1585],
         [-0.0522, -0.2915],
         [ 0.1668, -0.2892]],

        [[ 0.1785, -0.0362],
         [ 0.1284,  0.2839],
         [ 0.0294,  0.1193],
         [ 0.2587, -0.0475],
         [-0.0819,  0.0925]],

        [[ 0.0732, -0.1452],
         [-0.1823,  0.1533],
         [-0.3081,  0.3008],
         [-0.2510, -0.0101],
         [ 0.2480, -0.3062]],

        [[-0.1084,  0.1401],
         [-0.2372,  0.2106],
         [ 0.2891,  0.2375],
         [ 0.0591,  0.1321],
         [ 0.0452, -0.0190]],

        [[ 0.0772,  0.1155],
         [-0.2830,  0.1173],
         [ 0.0378, -0.2810],
         [ 0.0704,  0.1977],
         [-0.0269,  0.2413]],

        [[-0.2443, -0.2715],
         [ 0.0310,  0.0220],
         [-0.2393,  0.2968],
         [-0.0781, -0.1188],
         [ 0.2109,  0.1117]]], requires_grad=True)
>>> cnn.conv.weight.size()
torch.Size([7, 5, 2])
>>> hist
>>> cnn.conv.weight.size()
torch.Size([7, 5, 2])
>>> x
tensor([[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29],
         [30, 31, 32, 33, 34]]])
>>> %run minimalcnn
>>> cnn.conv.weight.size()
torch.Size([7, 5, 2])
>>> %run minimalcnn
>>> cnn.conv.weight.set_(torch.ones_like(cnn.conv.weight))
>>> torch.prod(cnn.conv.weight, torch.zeros_like(cnn.conv.weight))
>>> torch.prod(cnn.conv.weight, torch.tensor(torch.zeros_like(cnn.conv.weight)))
>>> torch.prod(torch.tensor(cnn.conv.weight), torch.tensor(torch.zeros_like(cnn.conv.weight)))
>>> type(cnn.conv.weight)
torch.nn.parameter.Parameter
>>> cnn.conv.weight.to_dense()
>>> cnn.conv.weight.to_dense
<function Parameter.to_dense>
>>> cnn.conv.weight.tensor_split()
>>> cnn.conv.weight.tensor_split
<function Parameter.tensor_split>
>>> cnn.conv.weight.as_strided()
>>> cnn.conv.weight.as_strided(1)
>>> cnn.conv.weight.as_strided((1,))
>>> cnn.conv.weight.as_strided(stride=1)
>>> help(cnn.conv.weight)
>>> cnn.conv.weight.super()
>>> cnn.conv.weight.tensor_split?
>>> cnn.conv.weight.tensor_split(0, 0)
>>> cnn.conv.weight.tensor_split(1, 0)
(tensor([[[-0.0799,  0.2315],
          [ 0.0009, -0.3020],
          [-0.2991, -0.0283],
          [-0.2625,  0.0761],
          [-0.1627, -0.1485]],
 
         [[-0.1528, -0.2559],
          [-0.2384,  0.2093],
          [-0.0730, -0.2590],
          [-0.1813,  0.1324],
          [-0.1357, -0.2571]],
 
         [[-0.2285,  0.0981],
          [-0.0911, -0.1366],
          [ 0.2790,  0.0272],
          [ 0.1797, -0.1473],
          [ 0.1371, -0.0307]],
 
         [[-0.1740, -0.0049],
          [ 0.0432,  0.0916],
          [ 0.0587, -0.0261],
          [ 0.0263,  0.0464],
          [-0.1956, -0.0487]],
 
         [[ 0.1742,  0.3146],
          [ 0.1122,  0.0179],
          [ 0.1781,  0.2864],
          [-0.0192, -0.2403],
          [ 0.2930, -0.0119]],
 
         [[-0.2497,  0.2714],
          [-0.0154, -0.0078],
          [-0.0857, -0.2081],
          [-0.1004, -0.2929],
          [ 0.2059,  0.0591]],
 
         [[ 0.1177,  0.2814],
          [-0.1974, -0.2281],
          [-0.0874, -0.1767],
          [-0.1300,  0.1230],
          [ 0.1271, -0.0973]]], grad_fn=<SliceBackward0>),)
>>> type(cnn.conv.weight.tensor_split(1, 0))
tuple
>>> cnn.conv.weight.data = torch.arange(torch.cumprod(cnn.conv.weight.data.size()))
>>> cnn.conv.weight.data = torch.arange(torch.cumprod(list(cnn.conv.weight.data.size())))
>>> cnn.conv.weight.data = torch.arange(torch.cumprod(torch.tensor(list(cnn.conv.weight.data.size()))))
>>> cnn.conv.weight.data = torch.arange(torch.cumprod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()])))
>>> torch.tensor([.1 * w for w in cnn.conv.weight.data.size()])
tensor([0.7000, 0.5000, 0.2000])
>>> cnn.conv.weight.data.size()
torch.Size([7, 5, 2])
>>> torch.cumprod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()]))
>>> torch.prod(torch.tensor([.1 * w for w in cnn.conv.weight.data.size()]))
tensor(0.0700)
>>> torch.prod(torch.tensor([w for w in cnn.conv.weight.data.size()]))
tensor(70)
>>> torch.prod(torch.tensor(cnn.conv.weight.data.size()))
tensor(70)
>>> torch.arange(torch.prod(torch.tensor(cnn.conv.weight.data.size())))
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
>>> torch.arange(torch.prod(torch.tensor(cnn.conv.weight.data.size())))
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
>>> w = _
>>> w.to(float)
tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
        28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41.,
        42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
        56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69.],
       dtype=torch.float64)
>>> cnn.conv.weight.data.set_(w.to(float))
>>> cnn.conv.weight.data.set_(w.to(float))
>>> w = w.to(float)
>>> w.resize_as(cnn.conv.weight.data)
tensor([[[ 0.,  1.],
         [ 2.,  3.],
         [ 4.,  5.],
         [ 6.,  7.],
         [ 8.,  9.]],

        [[10., 11.],
         [12., 13.],
         [14., 15.],
         [16., 17.],
         [18., 19.]],

        [[20., 21.],
         [22., 23.],
         [24., 25.],
         [26., 27.],
         [28., 29.]],

        [[30., 31.],
         [32., 33.],
         [34., 35.],
         [36., 37.],
         [38., 39.]],

        [[40., 41.],
         [42., 43.],
         [44., 45.],
         [46., 47.],
         [48., 49.]],

        [[50., 51.],
         [52., 53.],
         [54., 55.],
         [56., 57.],
         [58., 59.]],

        [[60., 61.],
         [62., 63.],
         [64., 65.],
         [66., 67.],
         [68., 69.]]], dtype=torch.float64)
>>> cnn.conv.weight.data.set_(w.resize_as(cnn.conv.weight.data))
>>> hist
>>> pwd
'/home/hobs/code/tangibleai/nlpia2/src/nlpia2/ch07'
>>> hist -o -p -f minimal-cnn.hist.md
