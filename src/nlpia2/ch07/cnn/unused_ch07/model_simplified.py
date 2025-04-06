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
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

#####################################################################
# .Compute the shape of the CNN output (the number of the output encoding vector dimensions)


def lopez_cnn_output_size(embedding_size, kernel_lengths, strides, desired_conv_output_size=None, **kwargs):
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


def compute_output_seq_len(input_seq_len, kernel_lengths, stride):
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
            (input_seq_len - 1 * (kernel_len - 1) - 1) // stride) + 1
        out_pool = ((out_conv - 1 * (kernel_len - 1) - 1) // stride) + 1
        out_pool_total += out_pool

    # return the len of a "flattened" vector that is passed into a fully connected (Linear) layer
    return out_pool_total

# .Compute the shape of the CNN output (the number of the output encoding vector dimensions)
##########################################################################


class CNNTextClassifier(nn.ModuleList):

    def __init__(self,
                 params=None,
                 win=False,
                 seq_len=35,
                 conv_output_size=32,
                 dropout_portion=.2,
                 kernel_lengths=[2],
                 stride=2,
                 embeddings=(2000, 50),
                 test_size=.1,
                 **kwargs):
        """ Conv1D layers concatenated into a single 1D vector

        python train.py --split_random_state=850753 --numpy_random_state=704 --torch_random_state=704463
        """

        super().__init__()
        if len(kwargs):
            log.warning(f"Did not process all kwargs: {kwargs}")
        self.random_state = kwargs.pop('random_state', None)
        if self.random_state is not None:
            self.torch_random_state = self.random_state
            self.numpy_random_state = self.random_state + 1
        if params.torch_random_state is None:
            self.torch_random_state = torch.random.initial_seed()
        else:
            self.torch_random_state = params.torch_random_state
        if params.numpy_random_state is None:
            self.numpy_random_state = np.random.get_state()[1][0]
        else:
            self.numpy_random_state = params.numpy_random_state

        try:
            shape = embeddings.shape
        except AttributeError:
            try:
                shape = embeddings.size()
            except AttributeError:
                shape = embeddings
        print(f'shape: {shape}')

        self.seq_len = 35  # seq_len
        self.vocab_size = shape[0]
        self.embedding_size = shape[1]
        self.num_groups = 1  # self.embedding_size
        self.num_input_channels = self.seq_len              # <1>
        self.kernel_lengths = [2]  # kernel_lengths         # <2>
        self.stride = 2
        # self.output_seq_len = compute_output_seq_len(   # <3>
        #     input_seq_len=self.seq_len,
        #     kernel_lengths=self.kernel_lengths,
        #     stride=self.stride,
        # )

        torch.random.manual_seed(self.torch_random_state)
        np.random.seed(self.numpy_random_state)

        self.output_seq_len = lopez_cnn_output_size(   # <3>
            embedding_size=self.embedding_size,
            kernel_lengths=self.kernel_lengths,
            strides=[self.stride] * len(kernel_lengths),
        )
        self.num_output_channels = self.output_seq_len

        assert self.torch_random_state == torch.random.initial_seed()
        assert self.numpy_random_state == np.random.get_state()[1][0]

        self.convolvers = []
        self.poolers = []

        # self.seq_len = params.seq_len
        # self.vocab_size = params.vocab_size
        print(f"self.embedding_size: {self.embedding_size}")
        self.embedding_size = params.embedding_size
        print(f"self.embedding_size: {self.embedding_size}")
        self.kernel_lengths = list(params.kernel_lengths)

        if isinstance(embeddings, torch.Tensor):
            print(f'Loading embeddings: {embeddings.size()}')
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            print(f'Creating empty embeddings: {self.vocab_size, self.embedding_size}')
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)

        for i, kernel_len in enumerate(self.kernel_lengths):
            self.convolvers.append(nn.Conv1d(
                in_channels=self.num_input_channels,
                out_channels=self.num_output_channels,
                kernel_size=kernel_len,
                groups=self.num_groups,
                stride=self.stride))
            self.poolers.append(nn.MaxPool1d(kernel_len, self.stride))

        self.dropout_portion = params.dropout_portion
        self.dropout = nn.Dropout(self.dropout_portion)

        print(f"conv_output_size: {conv_output_size}")

        self.linear_layer = nn.Linear(self.num_output_channels, 1)


# <1> assume a maximum text length of 35 tokens
# <2> only one kernel layer is needed for reasonable results
# <3> the convolution output need not have the same number of channels as your embeddings


    def forward(self, x):
        """ Takes sequence of integers (token indices) and outputs binary class label """

        x = self.embedding(x)

        conv_outputs = []
        for (conv, pool) in zip(self.convolvers, self.poolers):
            z = conv(x)
            z = torch.relu(z)
            z = pool(z)
            conv_outputs.append(z)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat(conv_outputs, 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.linear_layer(union)
        # Dropout is applied
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)

        return out.squeeze()
