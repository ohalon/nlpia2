""" word2vec trained on Wikipedia text with pytorch
## Resources

```yaml
-
  citation: Mikolov et al (2013) ARXIV Computation and Language
  url: https://arxiv.org/abs/1301.3781)
-
  citation: Musashi (Jacobs-) Harukawa (21 Oct 2021) Personal Blog
  description: Word2Vec from scratch on tweet hate speech
  url: https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html
-
  citation: Garg et al (2018), PNAS
  description: shifting stereotypes over 100 yrs
  url: https://www.pnas.org/content/pnas/115/16/E3635.full.pdf
-
  citation: Rodman (2019), Political Analysis
  description: meaning of "equality" in US
  url: https://static1.squarespace.com/static/5ca7d04ea09a7e68ba44e707/t/5cda219af4e1fc94236bc0cf/1557799325771/Diachronic_Word_Vectors___Political_Analysis_Final_Version.pdf # noqa
-
  citation: Rodriguez and Spirling (2021), Journal of Politics
  description: word embeddings for social science
  url: https://www.journals.uchicago.edu/doi/10.1086/715162
```
"""
from collections import Counter
from itertools import chain
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MIN_FREQ = 10  # ignore words that appear fewer than 10 times
WINDOW_WIDTH = 3

# ~1.8M lines of wikipedia text: https://pytorch.org/text/stable/datasets.html#wikitext-103
# dset = datasets.Wikitext103()

# ~40k lines (paragraphs) of Wikipedia text: https://pytorch.org/text/stable/datasets.html#wikitext-2
# Short and empty lines (nonsentences) are not filtered out
dsets = datasets.WikiText2()

pattern = r"\w+(?:\'\w+)?|[A-Z][a-z0-9]+|[^\w\s]+"
re_pattern = re.compile(pattern)


# chain all 3 sample sets in dset[0] (train), dset[1] (validation), dset[2] (test)
tokenized_lines = map(re_pattern.findall, chain(*dsets))

counts = Counter(chain(*tokenized_lines))
print(len(counts))
# 33153

counts = {k: v for k, v in counts.items() if v > MIN_FREQ}  # ignore rare terms
vocab = list(counts.keys())
# pd.Series()
vocab_size = len(vocab)
id2tok = dict(enumerate(vocab))
tok2id = {token: id for id, token in id2tok.items()}

len(counts)
# 15047


def delete_oov(dset):
    return [[tok for tok in line if tok in vocab] for line in dset]


def neighbor_pairs(tokens, window_width=WINDOW_WIDTH):
    """ Mikolov called these skip grams - pairs of words that are within window_width of each other

    >>> neighbor_pairs('Every good boy does fine .'.split(), 3)
    [
     (Every, good),
     (Every, boy),
     (Every, does),
     (good, boy),
     (good, does),
     (good, fine),
    ...
    """
    pairs = []
    for i, wd in enumerate(tokens):
        target = tok2id[wd]
        window = [
            i + j for j in
            range(-window_width, window_width + 1, 1)
            if (i + j >= 0)
            & (i + j < len(tokens))
            & (j != 0)
        ]

        pairs.extend([(target, tok2id[tokens[w]]) for w in window])
    # huggingface datasets are dictionaries for every text element
    # line['moving_window'] = pairs
    # return line
    return pairs


###################################################
# dataloader


class Word2VecDataset(Dataset):
    """
    Takes a PyTorch Dataset as an input, to be used for a Word2Vec dataloader.
    """

    # FIXME: https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html
    # FIXME: original blog used dataset['moving_window'] in place of pairs
    def __init__(self, pairs, vocab_size):  # , window_width=WINDOW_WIDTH):
        self.dataset = pairs
        self.vocab_size = vocab_size
        self.data = np.array([i for s in pairs for i in s])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


BATCH_SIZE = 2**14
N_LOADER_PROCS = 10


dsets = datasets.WikiText2()
dataloader = {}
# Convert pytorch raw text datasets (3-tuple) to a dict with 3 key-value pairs for train/val/test
dataset = {'train': list(dsets[0]), 'validation': list(dsets[1]), 'test': list(dsets[2])}

for key in dataset.keys():
    dataloader = {
        key: DataLoader(
            Word2VecDataset(dataset[key], vocab_size=vocab_size),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=N_LOADER_PROCS
        )
    }

# dataloader
###################################################


def one_hot_encode(token_id, vocab_size):
    vec = torch.zeros(vocab_size).float()
    vec[token_id] = 1.0
    return vec


ohe = one_hot_encode(token_id=3, vocab_size=10)
linear_layer = nn.Linear(vocab_size, 1, bias=False)

# Set edge weights from 0 to 9 for easy reference
with torch.no_grad():
    linear_layer.weight = nn.Parameter(
        torch.arange(10, dtype=torch.float).reshape(linear_layer.weight.shape))

print(linear_layer.weight)
# Parameter containing:
# tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]], requires_grad=True)

print(linear_layer(ohe))
# tensor([3.], grad_fn=<SqueezeBackward3>)


embedding_layer = nn.Embedding(vocab_size, 1)

with torch.no_grad():
    embedding_layer.weight = nn.Parameter(
        torch.arange(10, dtype=torch.float
                     ).reshape(embedding_layer.weight.shape))

print(embedding_layer.weight)
print(embedding_layer(torch.tensor(input)))


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        # Encode input to lower-dimensional representation
        hidden = self.embed(input)
        # Expand hidden layer to predictions
        logits = self.expand(hidden)
        return logits


# Instantiate the model
EMBED_SIZE = 100  # Mikalov found that 300-D word vectors worked well
model = Word2Vec(vocab_size, EMBED_SIZE)

# Relevant if you have a GPU:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define training parameters
LR = 3e-4
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


pbar = tqdm(range(EPOCHS * len(dataloader['train'])))
running_loss = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for center, context in dataloader['train']:
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad()
        logits = model(input=context)
        loss = loss_fn(logits, center)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.update(1)
    epoch_loss /= len(dataloader['train'])
    running_loss.append(epoch_loss)
