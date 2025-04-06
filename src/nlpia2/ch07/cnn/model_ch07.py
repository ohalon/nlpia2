""" 1-D Convolutional Neural Network for NLP

FIXME: Verify predict and compute_accuracy() functions by comparing to older versions in git

Definitions (from PyTorch Docs):
    in_channels: Number of channels in the input image/sequence
    out_channels: Number of channels produced by the convolution (encoding vector length)
    kernel_size: Size of the convolving kernel
    stride: Stride of the convolution. Default: 1
    padding: Padding added to both sides of the input. Default: 0
    padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation: Spacing between kernel elements. Default: 1
    groups: Number of blocked connections from input channels to output channels. Default: 1
    bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

$ python main.py
Epoch: 1, loss: 0.71129, Train accuracy: 0.56970, Test accuracy: 0.64698
...
Epoch: 10, loss: 0.38202, Train accuracy: 0.80324, Test accuracy: 0.75984
"""
import logging
import numpy as np  # noqa
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

#####################################################################
# .Compute the shape of the CNN output (the number of the output encoding vector dimensions)


def lopez_cnn_output_size(embedding_size, kernel_lengths, strides, desired_conv_output_size=None):
    """ Calculate the number of encoding dimensions output from CNN layers

    Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    if desired_conv_output_size is None:
        desired_conv_output_size = embedding_size // 2
    out_pool_total = 0
    for kernel_len, stride in zip(kernel_lengths, strides):
        out_conv = ((embedding_size - 1 * (kernel_len - 1) - 1) // stride) + 1
        out_pool = ((out_conv - 1 * (kernel_len - 1) - 1) // stride) + 1
        out_pool_total += out_pool

    # Returns "flattened" vector (input for fully connected layer)
    return out_pool_total * desired_conv_output_size


def compute_output_seq_len(embedding_size=50, kernel_lengths=[2], stride=1, **kwargs):
    """ Calculate the number of encoding dimensions output from CNN layers

    From PyTorch docs:
      L_out = 1 + (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    But padding=0 and dilation=1, because we're only doing a 'valid' convolution.
    So:
      L_out = 1 + (L_in - (kernel_size - 1) - 1) // stride

    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    out_pool_total = 0
    for kernel_len in kernel_lengths:
        out_conv = (
            (embedding_size - (kernel_len - 1) - 1) // stride) + 1
        out_pool = ((out_conv - (kernel_len - 1) - 1) // stride) + 1
        out_pool_total += out_pool

    # return the len of a "flattened" vector that is passed into a fully connected (Linear) layer
    return out_pool_total

# .Compute the shape of the CNN output (the number of the output encoding vector dimensions)
##########################################################################


# .CNN hyperparameters
# [source,python]
# ----
class CNNTextClassifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.seq_len = 35                          # <1>
        self.vocab_size3000                        # <2>
        self.embedding_size = 50                   # <3>
        self.kernel_lengths = [2]                  # <4>
        self.stride = 1                            # <5>
        self.dropout = nn.Dropout(.2)              # <6>

        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0)

        self.conv_output_size = 35  # self.embedding_size
        self.convolvers = []
        self.poolers = []
        for i, kernel_size in enumerate(self.kernel_lengths):
            self.convolvers.append(
                nn.Conv1d(in_channels=34,
                          out_channels=34,
                          kernel_size=kernel_size,
                          stride=self.stride))
            self.poolers.append(
                nn.MaxPool1d(kernel_size, self.stride))  # <7>

        # self.conv_output_size = lopez_cnn_output_size()  # <8>
        self.conv_output_size = compute_output_seq_len(
            embedding_size=self.embedding_size,
            kernel_lengths=self.kernel_lengths,
            stride=1)  # <8>
        self.linear_layer = nn.Linear(self.conv_output_size, 1)
# ----
# <1> `N_`: assume a maximum text length of 35 tokens
# <2> `V`: number of unique tokens (words) in your vocabulary
# <3> `E`: number of dimensions in your word embeddings
# <4> `K`: number of weights in each kernel
# <5> `S`: number of time steps (tokens) to slide the kernel forward with each step
# <6> `D`: portion of convolution output to
# <7> `P`: each convolution layer gets its own pooling function
# <8> `C`: the total convolutional output size depends on how many and what shape convolutions you choose

# .Stacking the CNN layers
# [source,python]
# ----
    def forward(self, x):
        """ Takes sequence of integers (token indices) and outputs binary class label """

        x = self.embedding(x).transpose(1, 2)

        conv_outputs = []
        for (conv, pool) in zip(self.convolvers, self.poolers):
            print(f"x.size(): {x.size()}")
            z = conv(x)
            print(f"conv(x).size(): {z.size()}")
            z = torch.relu(z)  # <1>
            print(f"conv(x).size().relu(): {z.size()}")
            z = pool(z)        # <2>
            print(f"pool(relu(conv(x).size())): {z.size()}")
            conv_outputs.append(z)

        cat = torch.cat(conv_outputs, 2)    # <3>
        print(f"cat: {cat.size()}")
        enc = cat.reshape(cat.size(0), -1)  # <4>
        print(f"enc: {enc.size()}")

        sparse_enc = self.dropout(enc)       # <5>
        print(f"sparse_enc: {sparse_enc.size()}")

        # FIXME: .linear(input, self.weight, self.bias)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x1155 and 48x1)

        out = self.linear_layer(sparse_enc)  # <6>
        out = torch.sigmoid(out)

        return out.squeeze()
# <1> each convolution layer gets its own activation function
# <2> each convolution layer gets a pooling function
# <3> concatenate the pooling outputs to create a single vector
# <4> flatten the output tensor to create an encoding vector
# <5> dropout (zero out) some encoding dimensions to make it sparse
# <6> output a linear (weighted) combination of the encoding values
# <7> for a binary class squash the output between 0 and 1
