import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_VOCAB_SIZE = 33278  # len(Corpus('./data/WikiText-2').dictionary)
DEVICE = torch.device('cpu')


# class RNNBaseModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder.

#     RNN Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two RNNs together to form a `stacked RNN`,
#             with the second RNN taking in outputs of the first RNN and
#             computing the final results. Default: 1
#         nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         embedding_dropout: If non-zero, introduces a `Dropout` layer on the outputs of the Embedding layer,
#         rnn_dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             RNN layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0.2
#     bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
#     """

#     def __init__(self,
#                  rnn_type='LSTM',
#                  input_size=200, hidden_size=200, num_layers=2, dropout=0.2,
#                  ntoken=DEFAULT_VOCAB_SIZE, tie_weights=False, **kwargs):
#         super().__init__()
#         self.rnn_type = rnn_type
#         self.activation_type = rnn_type.split('_')[-1] if '_' in rnn_type else None
#         self.ntoken = ntoken
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, input_size)
#         self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
#         self.decoder = nn.Linear(hidden_size, ntoken)

#         # Share weights for input and output token embeddings:
#         # ["Using the Output Embedding to Improve Language Models", Press & Wolf 2016](https://arxiv.org/abs/1608.05859)
#         # ["Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling", Inan et al. 2016](https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             if hidden_size != input_size:
#                 raise ValueError('When using the tied flag, hidden_size must be equal to emsize')
#             self.decoder.weight = self.encoder.weight

#         self.init_weights()

#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, input, hidden):
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(output)
#         decoded = decoded.view(-1, self.ntoken)
#         return F.log_softmax(decoded, dim=1), hidden

#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         if self.rnn_type == 'LSTM':
#             return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
#                     weight.new_zeros(self.num_layers, bsz, self.hidden_size))
#         else:
#             return weight.new_zeros(self.num_layers, bsz, self.hidden_size)


# class LSTMModel(RNNBaseModel):
#     """Container module with an encoder, a recurrent module, and a decoder."""

#     def __init__(self, rnn_type='LSTM', input_size=200, hidden_size=200, num_layers=2, dropout=0.2, ntoken=DEFAULT_VOCAB_SIZE, tie_weights=False):
#         super().__init__()
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.bias)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, input, hidden):
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(output)
#         decoded = decoded.view(-1, self.ntoken)
#         return F.log_softmax(decoded, dim=1), hidden

#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         if self.rnn_type == 'LSTM':
#             return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
#                     weight.new_zeros(self.num_layers, bsz, self.hidden_size))
#         else:
#             return weight.new_zeros(self.num_layers, bsz, self.hidden_size)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.

    TODO: Create RNNBaseModel parent and child RNNModel, GRUModel, LSTMModel classes
    """

    def __init__(self, rnn_type, vocab, nonlinearity='RELU', input_size=200,
                 hidden_size=200, batch_size=20, num_layers=2,
                 embedding_dropout=0, rnn_dropout=0.2,
                 bidirectional=False,
                 tie_weights=False, **kwargs):
        super().__init__()
        self.rnn_type = rnn_type.split('_')[0].upper()
        self.nonlinearity = str.lower(nonlinearity or rnn_type.split('_')[-1])
        self.vocab = vocab
        self.input_size = input_size
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn_dropout = embedding_dropout
        self.encoder = nn.Embedding(len(self.vocab), self.input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        rnn_kwargs = dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout,
        )
        if self.rnn_type == 'RNN':
            rnn_kwargs['nonlinearity'] = self.nonlinearity or 'relu'
        self.rnn = getattr(nn, self.rnn_type)(**rnn_kwargs)
        self.decoder = nn.Linear(hidden_size, len(self.vocab))

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != input_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # Why aren't the RNN weights initialized? Are they the same ones as in init_hidden()?
        # what would this do: `nn.init.uniform_(self.rnn.weight, -initrange, initrange)`
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.embedding_dropout(emb)
        output, hidden = self.rnn(emb, hidden)
        # output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, len(self.vocab))
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (
                weight.new_zeros(
                    self.num_layers, self.batch_size, self.hidden_size),
                weight.new_zeros(
                    self.num_layers, self.batch_size, self.hidden_size)
            )
        else:
            return weight.new_zeros(
                self.num_layers, self.batch_size, self.hidden_size)

    def predict_word_hidden(self, input_word, hidden_tens=None, temperature=1, device=None):
        if hidden_tens is None:
            hidden_tens = self.init_hidden()
        device = device or hidden_tens.device
        input_tens = torch.LongTensor(
            [[self.vocab.word2idx[input_word]]]).to(device)
        output_tens, hidden_tens = self(input_tens, hidden_tens)
        word_weights = output_tens.squeeze().div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        # if word_idx == IDX_UNK:
        #     continue
        input_tens.fill_(word_idx)
        return self.vocab.idx2word[word_idx], hidden_tens
