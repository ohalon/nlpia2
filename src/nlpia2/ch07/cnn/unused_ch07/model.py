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


def calc_output_seq_len(in_seq_len, kernel_lengths, strides, dilation=1):
    """ Calculate the number of encoding dimensions output from a concatenated CNN

    From PyTorch docs:
      L_out = 1 + (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    But padding=0 and dilation=1, because we're only doing a 'valid' convolution.
    So:
      L_out = 1 + (L_in - (kernel_size - 1) - 1) // stride

    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    if isinstance(strides, int):
        strides = [strides] * len(kernel_lengths)
    out_pool_total = 0
    for i, (kernel_len, stride) in enumerate(zip(kernel_lengths, strides)):
        out_conv = (
            (in_seq_len - dilation * (kernel_len - 1) - 1) // stride) + 1
        log.warning(f"out_conv[{i}]({kernel_len}, {stride}): {out_conv}")
        out_pool = (
            (out_conv - dilation * (kernel_len - 1) - 1) // stride) + 1
        log.warning(f"out_pool[{i}]: {out_pool}")
        out_pool_total += out_pool
        log.warning(f"out_pool_total[{i}]: {out_pool_total}")

    # return the len of a "flattened" vector that is passed into a fully connected (Linear) layer
    out_pool_total = out_pool_total * in_seq_len // (sum(strides) // len(strides))
    log.warning(f"out_pool_total (FINAL return value): {out_pool_total}")
    return out_pool_total

# .Compute the shape of the CNN output (the number of the output encoding vector dimensions)
##########################################################################


class CNNTextClassifier(nn.ModuleList):

    def __init__(self,
                 random_state=None,
                 torch_random_state=None,
                 numpy_random_state=None,

                 win=False,
                 seq_len=32,
                 in_channels=50,
                 out_channels=50,
                 dropout_portion=.15,
                 kernel_lengths=[2],
                 groups=None,
                 stride=2,
                 strides=None,
                 embeddings=(2000, 64),
                 test_size=.1,
                 **kwargs):
        """ Conv1D layers concatenated into a single 1D vector

        python train.py --split_random_state=850753 --numpy_random_state=704 --torch_random_state=704463
        """
        super().__init__()
        if len(kwargs):
            log.warning(f"Did not process all kwargs: {kwargs}")

        self.first_time = True
        self.random_state = random_state
        if self.random_state is not None:
            self.torch_random_state = self.random_state
            self.numpy_random_state = self.random_state + 1
        if torch_random_state is None:
            self.torch_random_state = torch.random.initial_seed()
        else:
            self.torch_random_state = torch_random_state
        if numpy_random_state is None:
            self.numpy_random_state = np.random.get_state()[1][0]
        else:
            self.numpy_random_state = numpy_random_state

        print(f"embeddings={embeddings}")
        try:
            shape = embeddings.shape
        except AttributeError:
            try:
                shape = embeddings.size()
            except AttributeError:
                shape = embeddings
        print(f'shape: {shape}')
        self.vocab_size = shape[0]
        self.embedding_size = shape[1]

        self.seq_len = seq_len
        self.in_channels = in_channels             # <1> seq_len
        self.out_channels = self.in_channels
        self.groups = groups
        self.kernel_lengths = [2]  # kernel_lengths         # <2>
        self.stride = 2
        self.strides = strides
        if self.strides is None or not len(self.strides) == len(self.kernel_lengths):
            self.strides = [self.stride] * len(self.kernel_lengths)
        # self.output_seq_len = compute_output_seq_len(   # <3>
        #     input_seq_len=self.seq_len,
        #     kernel_lengths=self.kernel_lengths,
        #     stride=self.stride,
        # )

        torch.random.manual_seed(self.torch_random_state)
        np.random.seed(self.numpy_random_state)

        assert self.torch_random_state == torch.random.initial_seed()
        assert self.numpy_random_state == np.random.get_state()[1][0]

        self.convolvers = []
        self.poolers = []

        print(f"self.embedding_size: {self.embedding_size}")
        self.kernel_lengths = list(kernel_lengths)

        if isinstance(embeddings, torch.Tensor):
            print(f'Loading embeddings: {embeddings.size()}')
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            print(f'Creating empty embeddings: {self.vocab_size, self.embedding_size}')
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)

        self.strides = strides
        if not self.strides:
            self.strides = [self.stride] * len(self.kernel_lengths)
        if len(self.strides) < len(self.kernel_lengths):
            self.strides = list(self.strides) + [self.stride] * (len(self.kernel_lengths) - len(self.strides))

        self.dropout_portion = dropout_portion
        self.dropout = nn.Dropout(self.dropout_portion)

        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_size, padding_idx=0)

        # default: 4 CNN layers with max pooling
        for i, (kernel_size, stride) in enumerate(zip(self.kernel_lengths, self.strides)):
            convkwargs = dict(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=self.groups,
            )
            print(f"Conv1d(kwargs={convkwargs})")
            self.convolvers.append(nn.Conv1d(
                **convkwargs))
            print(self.convolvers[-1])
            self.poolers.append(nn.MaxPool1d(kernel_size, stride))
            print(f"self.poolers[-1]: {self.poolers[-1]}")

        calcoutkwargs = dict(in_seq_len=self.in_channels * 2,  # seq_len
                             kernel_lengths=self.kernel_lengths,
                             strides=self.strides)
        print(calcoutkwargs)
        self.encoding_size = calc_output_seq_len(
            **calcoutkwargs,
        )
        print(f"self.encoding_size = {self.encoding_size}")

        self.linear_layer = nn.Linear(self.encoding_size, 1)
# <1> assume a maximum text length of 32 tokens
# <2> only one kernel layer is needed for reasonable results
# <3> the convolution output need not have the same number of channels as your embeddings

    def forward(self, x):
        """ Input is sequence of ints (token indices). Output is a single binary class label """

        X = self.embedding(x)
        conv_outputs = []
        for (conv, pool) in zip(self.convolvers, self.poolers):
            z = conv(X)
            z = torch.relu(z)
            z = pool(z)
            conv_outputs.append(z)

        if self.first_time:
            print(f"conv_outputs: {[co.size() for co in conv_outputs]}")
        encoding = torch.cat(conv_outputs, 2)
        if self.first_time:
            print(f"encoding.size(): {encoding.size()}")
        encoding = encoding.reshape(encoding.size(0), -1)
        if self.first_time:
            print(f"reshaped encoding.size(): {encoding.size()}")

        encoding = self.dropout(encoding)
        predictions = self.linear_layer(encoding)
        predictions = torch.sigmoid(predictions)

        self.first_time = False
        return predictions.squeeze()
