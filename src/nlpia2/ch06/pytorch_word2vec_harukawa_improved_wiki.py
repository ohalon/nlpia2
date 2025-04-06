from pathlib import Path
import csv

import pandas as pd
import torch
import datasets
import torchtext

from nessvec.constants import DATA_DIR  # noqa


BATCH_SIZE = 2**14  # 16384
BATCH_SIZE = 2**12  # 4096
CPU_CORES = 8


# tweetdataset = datasets.load_dataset('tweets_hate_speech_detection')
dsets = torchtext.datasets.WikiText2()


num_texts = 10000
filepaths = []
for name, dset in zip('train validation test'.split(), dsets):
    df = pd.DataFrame(dsets[0], columns=['text'])
    # df['idx'] = range(len(df))
    df['label'] = 1
    filepaths.append(str(DATA_DIR / f'WikiText2-{name}'))
    df.to_csv(filepaths[-1] + '.csv', index=False, quoting=csv.QUOTE_ALL)
    df.to_json(filepaths[-1] + '.json')  # , orient='records')
    with open(filepaths[-1] + '.txt', 'wt') as fout:
        fout.writelines(df['text'])
    with open(filepaths[-1] + f'.{num_texts}.txt', 'wt') as fout:
        fout.writelines(df['text'][:num_texts])

# dset = datasets.load_dataset('csv', data_files=filepaths[0] + '.csv')
# dset = datasets.load_dataset('json', data_files=filepaths[0] + '.json')
dset = datasets.load_dataset('text', data_files=filepaths[0] + f'.{num_texts}.txt')


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
                                         row['text'].lower()))  # 1
                         if (i not in sw) and len(i)]  # 4
    return row


# tokenizer with integrated stemmer, and stopword filter
#######################################################


################################################################
# count token frequency so rare tokens can be filtered, also stem all words

# Determine vocabulary so we can create mapping
dset = dset.map(split_tokens)


from collections import Counter  # noqa

counts = Counter([i for s in dset['train']['all_tokens'] for i in s])
counts = {k: v for k, v in counts.items() if v > 10}  # Filtering
vocab = list(counts.keys())
n_v = len(vocab)
id2tok = dict(enumerate(vocab))
tok2id = {token: id for id, token in id2tok.items()}

# Now correct tokens


def remove_rare_tokens(row):
    row['tokens'] = [t for t in row['all_tokens'] if t in vocab]
    return row


dset = dset.map(remove_rare_tokens)

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


dset = dset.map(windowizer)


######################################################################
# Better Windowizer

WINDOW_WIDTH = 3


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

#
################################################################


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


dataloader = {}
for k in dset.keys():
    dataloader = {
        k: DataLoader(
            Word2VecDataset(
                dset[k],
                vocab_size=n_v),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=CPU_CORES - 1)
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
LR = 5e-4
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# NN model
################################################


################################################
# Save (persist) NN model


def save_model(model, loss=torch.tensor(float("NaN"))):
    try:
        loss = loss.item()
    except AttributeError:
        pass

    model_dir = Path(DATA_DIR) / 'models'
    model_dir.mkdir(exist_ok=True)

    num_embeddings, embedding_dim = model.state_dict()["embed.weight"].shape
    print(num_embeddings, embedding_dim == model.state_dict()["expand.weight"].shape)

    import datetime  # noqa
    now = datetime.datetime.now()

    # filename = f'Word2Vec-state_dict-{now.isoformat()[:13]}.pt'
    # filename = f'Word2Vec-state_dict-{num_embeddings}x{embedding_dim}+{embedding_dim}x{num_embeddings}.pt'
    filename = f'Word2Vec-state_dict-{now.isoformat()[:13]}-loss_{loss}.pt'

    torch.save(model.state_dict(), model_dir / filename)
    return model_dir / filename


# NN model
################################################


################################################
# Training

from tqdm import tqdm  # noqa
running_loss = []

pbar = tqdm(range(EPOCHS * len(dataloader['train'])))
for epoch in range(EPOCHS):
    epoch_loss = 0
    for sample_num, (center, context) in enumerate(dataloader['train']):
        if sample_num % len(dataloader['train']) == 2:
            print(center, context)
            # center: tensor([ 229,    0, 2379,  ...,  402,  553,  521])
            # context: tensor([ 112, 1734,  802,  ...,   28,  852,  363])
        center, context = center.to(device), context.to(device)
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

save_model(model, loss)

# Training
################################################


################################################
# Learning curve plot

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
plt.show(block=False)


###################################################
# Vectors = embedding layer model weights

wordvecs = model.expand.weight.cpu().detach().numpy()
tokens = ['father', 'mother', 'school', 'women', 'men', 'black', 'nice', 'kind', 'cool', 'mean', 'meaning']

# Vectors = embedding layer model weights
###################################################


###################################################
# Evaluation (word semantic similarity)


from scipy.spatial import distance  # noqa
import numpy as np  # noqa


def get_distance_matrix(wordvecs, metric):
    dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
    return dist_matrix


def get_k_similar_words(word, dist_matrix, k=10):
    idx = tok2id[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k + 1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, id2tok[i], dists[i]) for i in ind]
    return out


dmat = get_distance_matrix(wordvecs, 'cosine')
for word in tokens:
    print(word, [t[1] for t in get_k_similar_words(word, dmat)], "\n")
# father ['dad', '#fathersday', 'day', 'wish', 'happi', 'love', 'enjoy', 'today', 'hope', 'wonder']
# mother ['children', 'ten', 'shit', 'nake', 'pic', 'issu', 'men', 'teen', 'nation', 'porn']
# school ['back', 'week', 'tomorrow', 'got', 'home', 'next', 'see', 'come', 'look', 'year']
# women ['black', 'fuck', 'men', 'girl', 'would', 'white', 'video', 'say', 'call', 'man']
# men ['women', 'black', 'hes', 'girl', 'white', 'guy', 'porn', 'sex', 'fuck', 'proud']
# black ['women', 'white', 'video', 'new', 'guy', 'girl', 'watch', 'say', 'like', 'call']
# hate ['would', 'peopl', 'much', 'mani', 'say', 'someon', 'still', 'dont', 'know', 'world']
# ass ['fuck', 'nigga', 'fan', 'shit', 'realli', 'win', 'shes', 'wont', 'black', 'game']
# nice ['morn', 'feel', 'friend', 'today', 'happi', 'like', 'amp', 'great', 'beauti', '@user']
# kind ['peopl', 'know', 'dont', 'sad', 'chang', 'even', 'world', 'let', 'one', 'someon']
# cool ['head', 'perfect', 'woman', 'stuff', 'found', 'joke', '#e', 'think', 'realli', 'new']
# mean ['sad', 'never', 'peopl', 'say', 'one', 'dont', 'even', 'would', 'cant', 'make']


# Evaluation: word semantic similarity
############
