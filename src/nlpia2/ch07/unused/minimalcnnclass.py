# !pip install dataframe-image
import dataframe_image as dfi

import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from matplotlib import pyplot as plt


class MinimalCNN(nn.Module):

    def __init__(self, kernel_size=2, stride=1, seq_len=5, vocab_size=11, embedding_size=7):
        super().__init__()

        # self.embedding = nn.Embedding(
        #     num_embeddings=5,
        #     embedding_dim=50,
        #     padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=embedding_size,
            # groups=50,
            kernel_size=kernel_size,
            stride=stride)
        print(f"self.conv.weights.size(): {self.conv.weight.size()}")
        print(f"self.conv.weights: {self.conv.weight}")

    def forward(self, x):
        # x = self.embedding(x)
        print(f"embeddings.size(): {x.size()}")
        print(f"embeddings:\n{x}")
        features = self.conv(x)
        print(f"features.size(): {features.size()}")
        print(f"features:\n{features}")
        return features.squeeze()
#             z = torch.relu(z)
#             z = pool(z)
#             conv_outputs.append(z)


if __name__ == "__main__":
    num_examples = 7
    seq_len = 5
    embedding_size = 1

    dataset = torch.arange(0, num_examples * seq_len * embedding_size, dtype=torch.float)
    dataset.resize_(num_examples, seq_len, embedding_size)

    x = torch.arange(
        num_examples * seq_len * embedding_size,
        dtype=torch.float
    )
    x.resize_(num_examples, seq_len, embedding_size)

    lin = nn.Linear(embedding_size * seq_len, 1)

    kernel_size = 2
    stride = 1

    cnn = MinimalCNN(
        stride=stride,
        kernel_size=kernel_size,
        seq_len=seq_len)
    print(cnn.conv.weight.size())
    cnn.forward(x)
