import torch
import datasets

dataset = datasets.load_dataset('tweets_hate_speech_detection')


#######################################################
# tokenizer with integrated stemmer, and stopword filter

# For simplicity let's remove alphanumeric but keep @, #
import re  # noqa
import nltk  # noqa
from nltk.corpus import stopwords  # noqa
from nltk.stem.snowball import SnowballStemmer  # noqa

ss = SnowballStemmer('english')

try:
    sw = stopwords.words('english')
except LookupError:
    nltk.download("stopwords")
    sw = stopwords.words('english')


def split_tokens(row):                             # STEP
    row['all_tokens'] = [ss.stem(i) for i in       # 5
                         re.split(r" +",               # 3
                                  re.sub(r"[^a-z@# ]", "",      # 2
                                         row['tweet'].lower()))  # 1
                         if (i not in sw) and len(i)]  # 4
    return row


# tokenizer with integrated stemmer, and stopword filter
#######################################################

# Determine vocabulary so we can create mapping
dataset = dataset.map(split_tokens)


################################################################
# count token frequency so rare tokens can be filtered, also stem all words

from collections import Counter  # noqa

counts = Counter([i for s in dataset['train']['all_tokens'] for i in s])
counts = {k: v for k, v in counts.items() if v > 10}  # Filtering
vocab = list(counts.keys())
n_v = len(vocab)
id2tok = dict(enumerate(vocab))
tok2id = {token: id for id, token in id2tok.items()}

# Now correct tokens


def remove_rare_tokens(row):
    row['tokens'] = [t for t in row['all_tokens'] if t in vocab]
    return row


dataset = dataset.map(remove_rare_tokens)

# count token frequency so rare tokens can be filtered, also stem all words
################################################################


def windowizer(row, wsize=3):
    """
    Windowizer function for Word2Vec. Converts sentence to sliding-window
    pairs.
    """
    doc = row['tokens']
    wsize = 3
    out = []
    for i, wd in enumerate(doc):
        target = tok2id[wd]
        window = [
            i + j for j in range(-wsize, wsize + 1, 1)
            if (i + j >= 0) & (i + j < len(doc)) & (j != 0)
        ]

        out += [(target, tok2id[doc[w]]) for w in window]
    row['moving_window'] = out
    return row


dataset = dataset.map(windowizer)


from torch.utils.data import Dataset, DataLoader  # noqa


class Word2VecDataset(Dataset):
    """
    Takes a HuggingFace dataset as an input, to be used for a Word2Vec dataloader.
    """

    def __init__(self, dataset, vocab_size, wsize=3):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.data = [i for s in dataset['moving_window'] for i in s]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


BATCH_SIZE = 2**14
N_LOADER_PROCS = 10

dataloader = {}
for key in dataset.keys():
    dataloader = {
        key: DataLoader(
            Word2VecDataset(
                dataset[key],
                vocab_size=n_v),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=N_LOADER_PROCS)
    }


################################################
# manual one-hot encoding (ohe)

from torch import nn  # noqa

size = 10
input = 3


def one_hot_encode(input, size):
    vec = torch.zeros(size).float()
    vec[input] = 1.0
    return vec


ohe = one_hot_encode(input, size)
linear_layer = nn.Linear(size, 1, bias=False)

# Set edge weights from 0 to 9 for easy reference
with torch.no_grad():
    linear_layer.weight = nn.Parameter(
        torch.arange(10, dtype=torch.float).reshape(linear_layer.weight.shape))

print(linear_layer.weight)
print(linear_layer(ohe))

# manual one-hot encoding (ohe)
################################################


################################################
# NN model

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
EMBED_SIZE = 100  # Quite small, just for the tutorial
model = Word2Vec(n_v, EMBED_SIZE)

# Relevant if you have a GPU:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define training parameters
LR = 3e-4
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


from tqdm import tqdm  # noqa

running_loss = []

progress_bar = tqdm(range(EPOCHS * len(dataloader['train'])))
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
        progress_bar.update(1)
    epoch_loss /= len(dataloader['train'])
    running_loss.append(epoch_loss)

import matplotlib.pyplot as plt  # noqa
plt.plot(running_loss)
plt.show(block=False)


# NN model
################################################


################################################
# Save (persist) NN model


from pathlib import Path  # noqa
from nessvec.constants import DATA_DIR  # noqa

model_dir = Path(DATA_DIR) / 'models'
model_dir.mkdir(exist_ok=True)

num_embeddings, embedding_dim = model.state_dict()["embed.weight"].shape
print(num_embeddings, embedding_dim == model.state_dict()["expand.weight"].shape)

import datetime  # noqa
now = datetime.datetime.now()

filename = f'Word2Vec-state_dict-{now.isoformat()[:13]}.pt'
filename = f'Word2Vec-state_dict-{num_embeddings}x{embedding_dim}+{embedding_dim}x{num_embeddings}.pt'
filename = f'Word2Vec-state_dict-{now.isoformat()[:13]}-loss_{loss.item()}.pt'

torch.save(model_dir / filename)

# NN model
################################################


################################################
# more verbose training


pbar = tqdm(range(EPOCHS * len(dataloader['train'])))
for epoch in range(EPOCHS):
    epoch_loss = 0
    for sample_num, (center, context) in enumerate(dataloader['train']):
        if not sample_num % len(dataloader['train']) == 2:
            print(center, context)
        center, context = center.to(device), context.to(device)
        if not sample_num % len(dataloader['train']) == 2:
            print(center, context)
        optimizer.zero_grad()
        logits = model(input=context)
        loss = loss_fn(logits, center)
        if not sample_num % 10000:
            # print(center, context)
            pbar.set_description(f'loss[{sample_num}] = {loss.item()}')
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.update(1)
    epoch_loss /= len(dataloader['train'])
    running_loss.append(epoch_loss)


# more verbose training
################################################


import seaborn as sns  # noqa
import matplotlib.pyplot as plt  # noqa
sns.set_theme()
plt.plot(running_loss)
plt.show(block=False)

sns.set_theme()
# sns.set_style('whitegrid')
# learning curve
plt.grid('on')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
