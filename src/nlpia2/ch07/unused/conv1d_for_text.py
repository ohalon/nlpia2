# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt


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
