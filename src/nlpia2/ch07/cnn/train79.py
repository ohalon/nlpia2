""" Includes random state args
FIXME: Verify predict and compute_accuracy() functions by comparing to older versions in git

$ python main.py
Epoch: 1, loss: 0.71129, Train accuracy: 0.56970, Test accuracy: 0.64698
...
Epoch: 10, loss: 0.38202, Train accuracy: 0.80324, Test accuracy: 0.75984
"""
import argparse
import sys
import time
from collections import Counter
import json
from itertools import chain
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import utils
from model79 import CNNTextClassifier
from nlpia2.language_model import nlp
import joblib


T0 = 1652404117  # number of seconds since 1970-01-01 as of May 12, 2022
MAX_SEED = 2**32 - 1
DATA_DIR = Path(__file__).parent / 'data'

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
log.setLevel(level=logging.INFO)


# # experiments/disaster_tweets_cnn_pipeline_24363.json  # May 29 16:12
# HYPERPARAMS_BEST = {
#     "expand_glove_vocab": False,
#     "epochs": 10,
#     "usecols": ["text", "target"],
#     "tokenizer": "tokenize_re",
#     "embeddings": [2000, 64],
#     "kernel_lengths": [2, 3, 4, 5],
#     "strides": [2, 2, 2, 2],
#     "conv_output_size": 32,
#     "in_channels": 32,
#     "planes": 1,
#     "out_channels": 32,
#     "groups": 1,
#     "batch_size": 12,
#     "learning_rate": 0.001,
#     "test_size": 0.1,
#     "dropout_portion": 0.2,
#     "num_stopwords": 0,
#     "case_sensitive": True,
#     "split_random_state": 1460940,
#     "numpy_random_state": 433,
#     "torch_random_state": 433994,
#     "re_sub": "[^A-Za-z0-9.?!]+",
#     "vocab_size": 2000,
#     "embedding_size": 64,
#     #    "learning_curve": [],
#     "loss": 0.11444409191608429,
#     "train_accuracy": 0.8727193110494819,
#     "test_accuracy": 0.7900262467191601,
# }
# # lopez example transposes the sequence of embeddings
# #   so that CNN channels for lopez are the tokens
# #   and time and CNN time/sequence/tokens are the embedding dimensions
# HYPERPARAMS_BEST["seq_len"] = HYPERPARAMS_BEST["in_channels"]


def tokenize_spacy(doc):
    return [tok.text for tok in nlp(doc) if tok.text.strip()]


def tokenize_re(doc):
    return [tok for tok in re.findall(r'\w+', doc)]


class Parameters(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath: Path = Path('disaster-tweets.csv')
        self.usecols: tuple = ('text', 'target')
        self.tokenizer: str = 'tokenize_re'

        self.epochs: int = 10

        self.embeddings: tuple = (2000, 64)
        self.kernel_lengths: list = [2, 3, 4, 5]
        self.strides: list = [2, 2, 2, 2]

        # lopez transposes the last 2 CNN tensor dimensions (time, in_channels).T => (in_channels, time)
        # lopez only works if in_channels and out_channels are 32 (equal)
        self.in_channels: int = 32
        self.seq_len = self.in_channels
        self.conv_output_size: int = 32

        self.planes: int = 1  # not sure if the is correct terminology
        self.out_channels: int = self.planes * self.in_channels
        self.groups: int = 1  # self.in_channels  # depth-first conv if groups == in_chan == out_channels / planes

        self.batch_size: int = 12
        self.learning_rate: float = 0.001
        self.test_size: float = 0.1

        self.dropout_portion: float = 0.2

        self.num_stopwords: int = 0
        self.case_sensitive: bool = True

        self.split_random_state: int = min(max(int((time.time() - T0)), 0), MAX_SEED)
        self.numpy_random_state: int = min(max(int((time.time() - T0 - self.split_random_state) * 1000), 0), MAX_SEED)
        self.torch_random_state: int = min(max(int((time.time() - T0 - self.split_random_state) * 1000000), 0), MAX_SEED)

        self.re_sub: str = r'[^A-Za-z0-9.?!]+'

        print(f"embeddings={self.embeddings}")
        try:
            shape = self.embeddings.shape
        except AttributeError:
            try:
                shape = self.embeddings.size()
            except AttributeError:
                shape = self.embeddings
        print(f'shape: {shape}')
        self.vocab_size = shape[0]
        self.embedding_size = shape[1]

        self.__names = 'filepath usecols tokenizer embeddings kernel_lengths'.split()
        self.__names += 'strides conv_output_size in_channels planes out_channels'.split()
        self.__names += 'groups epochs batch_size learning_rate test_size dropout_portion num_stopwords'.split()
        self.__names += 'case_sensitive split_random_state numpy_random_state torch_random_state'.split()

    def to_dict(self):
        for k in self.__names:
            self[k] = getattr(self, k)
        return self

    def parse_args(self):
        d = self.to_dict()
        self._parser = argparse.ArgumentParser(description='PyTorch CNN disaster tweet natural language text classifier.')
        for name, value in d.items():
            typ = type(value)
            self._parser.add_argument(f'--{name}', type=typ, default=value,
                                      help=f'{name}: {typ} (default = {value})')

        self.args = self._parser.parse_args()
        for k in d:
            print(k, getattr(self, k))

        return self


def update_params(params, **kwargs):
    # use "winning" train-test split to repro best results in NLPiA 2nd Ed
    if kwargs.pop('win', False):
        kwargs['split_random_state'] = 1460940  # seems to create easier testset
        kwargs['numpy_random_state'] = 1  # np.random not used so no effect
        kwargs['torch_random_state'] = 1  # moderate effect on best accuracy
    for param_name, param_val in params.__dict__.items():
        log.info(f'DEFAULT: {param_name}: {param_val}')
        kwarg_val = kwargs.get(param_name)
        if kwarg_val is not None:
            if param_val is None:
                coerce_to_dest_type = int
            else:
                coerce_to_dest_type = type(param_val)
            if not isinstance(param_val, str) and isinstance(kwarg_val, str):
                kwarg_val = eval(kwarg_val)
            setattr(params, param_name, coerce_to_dest_type(kwarg_val))
            log.warning(f'NEW KWARGS: {param_name}: {getattr(params, param_name)} ({type(getattr(params, param_name))})')
    params.tokenizer_fun = globals().get(params.tokenizer, tokenize_re)
    if not params.filepath.is_file():
        params.filepath = Path(DATA_DIR) / params.filepath
    return params


HYPERPARAMS = Parameters()
pipeline_args, pipeline_kwargs = utils.parse_argv(sys.argv)
HYPERPARAMS = update_params(params=HYPERPARAMS, **pipeline_kwargs)


def pad(sequence, pad_value=0, seq_len=HYPERPARAMS.seq_len):
    log.debug(f'BEFORE PADDING: {sequence}')
    padded = list(sequence)[:seq_len]
    padded = padded + [pad_value] * (seq_len - len(padded))
    log.debug(f'AFTER PADDING: {sequence}')
    return padded


def load_dataset(
        params,
        # seq_len=HYPERARAMS['seq_len'],
        vocab_size=2000,
        embedding_size=64,
        num_stopwords=0,
        test_size=HYPERPARAMS.test_size,
        expand_glove_vocab=False,
        **kwargs):
    """ load and preprocess csv file: return [(token id sequences, label)...]

    1. Simplified: load the CSV
    2. Configurable: case folding
    3. Configurable: remove non-letters (nonalpha):
    4. Configurable: tokenize with regex or spacy
    5. Configurable: remove stopwords (frequent words)
    6. Configurable: filter infrequent words
    7. Simplified: compute reverse index
    8. Simplified: transform token sequences to integer id sequences
    9. Simplified: pad token id sequences
    10. Simplified: train_test_split
    """
    params = update_params(params, **kwargs)
    split_random_state = params.split_random_state
    if split_random_state is None:
        split_random_state = kwargs.pop('split_random_state')
    split_random_state = int(split_random_state)

    log.warning(f'Using tokenizer={params.tokenizer_fun}')

    # 1. load the CSV
    df = pd.read_csv(params.filepath.open(), usecols=params.usecols)
    texts = df[params.usecols[0]].values
    targets = df[params.usecols[1]].values

    # 2. optional case folding:
    if not params.case_sensitive:
        texts = [str.lower(x) for x in texts]

    # 3. optional character (non-letter) filtering:
    texts = [re.sub(params.re_sub, ' ', x) for x in texts]

    # 4. customizable tokenization:
    texts = [params.tokenizer_fun(doc) for doc in tqdm(texts)]

    # 5. count frequency of tokens
    counts = Counter(chain(*texts))

    # 6. configurable num_stopwords and vocab_size
    vocab = [x[0] for x in counts.most_common(vocab_size + params.num_stopwords)]
    vocab = ['<PAD>'] + list(vocab[params.num_stopwords:])
    # id2tok = vocab

    # 7. compute reverse index
    tok2id = dict(zip(vocab, range(len(vocab))))

    # 8. Simplified: transform token sequences to integer id sequences
    id_sequences = [[i for i in map(tok2id.get, seq) if i is not None] for seq in texts]

    # 9. Simplified: pad token id sequences
    padded_sequences = []
    for s in id_sequences:
        padded_sequences.append(pad(s, pad_value=0, seq_len=params.seq_len))
    padded_sequences = torch.IntTensor(padded_sequences)

    # 10. Configurable sampling for testset (test_size samples)
    retval = params.to_dict()
    retval.update(dict(zip(
        'x_train x_test y_train y_test'.split(),
        train_test_split(
            padded_sequences,
            targets,
            test_size=test_size,
            random_state=split_random_state)
    )))
    retval['tokenizer_fun'] = params.tokenizer_fun
    retval['vocab_size'] = vocab_size
    retval['embedding_size'] = embedding_size
    retval['num_stopwords'] = num_stopwords
    retval['test_size'] = HYPERPARAMS.test_size
    retval['split_random_state'] = split_random_state
    retval['vocab'] = vocab
    retval['tok2id'] = tok2id
    return retval


class DatasetMapper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def calculate_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(y_true, y_pred):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives + true_negatives) / len(y_true)


class Pipeline(Parameters):

    def __init__(self, **kwargs):
        """ Tokenize and train-test split the disaster tweets then fit CNNTextClassifier"""
        super().__init__(**kwargs)
        log.info(kwargs)
        params = update_params(params=self)
        # self.__dict__.update(params.__dict__)
        self.verbose = int(float(kwargs.pop('verbose', 0) or 0))

        self.print(vars(self))

        dataset = load_dataset(params, **kwargs)

        self.epochs = int(float(dataset['epochs']))
        self.vocab = dataset['vocab']
        self.tokenizer_fun = dataset['tokenizer_fun']
        self.tok2id = dataset['tok2id']
        self.id2tok = self.vocab
        self.embedding_size = dataset['embedding_size']
        self.num_stopwords = int(float(dataset['num_stopwords']))
        self.test_size = dataset['test_size']
        self.split_random_state = int(float(dataset['split_random_state']))
        self.x_train = dataset['x_train']
        self.y_train = dataset['y_train']
        self.x_test = dataset['x_test']
        self.y_test = dataset['y_test']
        self.model = CNNTextClassifier(**params.__dict__)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def train(self, X=None, y=None):

        self.trainset_mapper = DatasetMapper(self.x_train, self.y_train)
        self.testset_mapper = DatasetMapper(self.x_test, self.y_test)

        self.loader_train = DataLoader(self.trainset_mapper, batch_size=self.batch_size)
        self.loader_test = DataLoader(self.testset_mapper, batch_size=self.batch_size)

        optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        self.learning_curve = []
        for epoch in range(self.epochs):
            self.model.train()
            predictions = []
            for x_batch, y_batch in self.loader_train:
                y_batch = y_batch.type(torch.FloatTensor)
                y_pred = self.model(x_batch)
                loss = F.binary_cross_entropy(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Save predictions
                predictions += list(y_pred.detach().numpy())

            test_predictions = self.predict()
            self.loss = loss.item()
            self.train_accuracy = calculate_accuracy(self.y_train, predictions)
            self.test_accuracy = calculate_accuracy(self.y_test, test_predictions)
            self.learning_curve += [[self.loss, self.train_accuracy, self.test_accuracy]]
            print(
                "Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f"
                % (epoch + 1, self.loss, self.train_accuracy, self.test_accuracy)
            )
        self.print([s for s in dir(self) if 'tok' in s.lower()])
        self.print([s for s in dir(self) if 'voc' in s.lower()])
        return self

    def indexes_to_texts(self, indexes):
        try:
            indexes = np.array(indexes)
        except:
            indexes = indexes.numpy()
        while len(indexes.shape) > 2:
            indexes = indexes[0]
        texts = []
        for row in indexes:
            texts.append(' '.join([self.id2tok[i] for i in row]))
        return texts

    def predict(self, X=None):

        self.model.eval()  # evaluation mode
        predictions = []

        if X is not None:
            X_batches = zip([X], [[None] * len(X)])
        else:
            X_batches = list(zip(*self.loader_test))[0]

        self.print(f'len(X_batches): {len(X_batches)}')
        self.print(f'len(X_batches[0]): {len(X_batches[0])}')
        self.print(f'len(X_batches[1]): {len(X_batches[1])}')
        self.print(f'X_batches[0].size(): {X_batches[0].size()}')
        self.print(f'X_batches[1].size(): {X_batches[1].size()}')
        self.print(f'X_batches[0][0]: {str(X_batches[0][0])[:80]}...')
        self.print(f'X_batches[0][1]: {str(X_batches[0][1])[:80]}...')
        # self.print('self.indexes_to_texts(X_batches[0][0])')
        # self.print(self.indexes_to_texts(X_batches[0][0]))
        # self.print('self.indexes_to_texts(X_batches[0][1])')
        # self.print(self.indexes_to_texts(X_batches[0][1]))

        with torch.no_grad():
            for x_batch in X_batches:
                y_pred = self.model(x_batch).detach().numpy()
                predictions += list(y_pred)
        return predictions

    def predict_text(self, X):
        if isinstance(X, str):
            X = [X]
        X_tokenized = [self.tokenizer_fun(s) for s in X]
        X = [[i for i in map(self.tok2id.get, toks) if i is not None] for toks in X_tokenized]
        X = torch.tensor(X)  # .to_device(self.device)
        self.print(X)
        return self.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y.detach.numpy())**2) ** .5

    def dump(self, filepath=None, indent=4):
        js = self.dumps(indent=indent)
        if filepath is None:
            t = int((time.time() - T0) / 60)
            filepath = f'disaster_tweets_cnn_pipeline_{t}.json'
        with open(filepath, 'w') as fout:
            fout.write(js)
        return js

    def dumps(self, indent=4):
        hashable_dict = {}
        for k, v in vars(self).items():
            if v is None or isinstance(v, (str, float, int, bool)):
                hashable_dict[k] = v
                continue
            if isinstance(v, (tuple, np.ndarray)):
                v = list(v)
            if isinstance(v, torch.Tensor):
                v = list(v.detach().numpy())
            if isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    v = [list(x.detach().numpy()) for x in v]
            try:
                hashable_dict[k] = json.loads(json.dumps(v))
            except TypeError:
                pass
        return json.dumps(hashable_dict, indent=indent)


def main(**pipeline_kwargs):
    pipeline = Pipeline(**pipeline_kwargs)

    pipeline = pipeline.train()
    hyperparams = json.loads(pipeline.dump())
    # predictions = pipeline.predict()

    return dict(pipeline=pipeline, hyperparams=hyperparams)


if __name__ == '__main__':
    """ Train a 1-D 4-kernel CNN on disaster-tweets.csv

    These CLI args achieved 79% accuracy once (but sometimes no better than 70%):

    ```bash
    $ python train.py \
        --conv_output_size=32 \
        --groups=1 \
        --embedding_size=50 \
        --in_channels=32 \
        --epochs=20 \
        --numpy_random_state=433 \
        --torch_random_state=433994 \
        --split_random_state=1460940

    WARNING: exact same accuracies achieved with different embedding size and in_channels


    Conv1d(kwargs={'in_channels': 32, 'out_channels': 32, 'kernel_size': 2, 'stride': 2, 'groups': 1})
    Conv1d(32, 32, kernel_size=(2,), stride=(2,))
    self.poolers[-1]: MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv1d(kwargs={'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'groups': 1})
    Conv1d(32, 32, kernel_size=(3,), stride=(2,))
    self.poolers[-1]: MaxPool1d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv1d(kwargs={'in_channels': 32, 'out_channels': 32, 'kernel_size': 4, 'stride': 2, 'groups': 1})
    Conv1d(32, 32, kernel_size=(4,), stride=(2,))
    self.poolers[-1]: MaxPool1d(kernel_size=4, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv1d(kwargs={'in_channels': 32, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'groups': 1})
    Conv1d(32, 32, kernel_size=(5,), stride=(2,))
    self.poolers[-1]: MaxPool1d(kernel_size=5, stride=2, padding=0, dilation=1, ceil_mode=False)
    {'in_seq_len': 64, 'kernel_lengths': [2, 3, 4, 5], 'strides': [2, 2, 2, 2]}
    self.encoding_size = 1856
    conv_outputs: [torch.Size([12, 32, 16]), torch.Size([12, 32, 15]), torch.Size([12, 32, 14]), torch.Size([12, 32, 13])]
    encoding.size(): torch.Size([12, 32, 58])
    reshaped encoding.size(): torch.Size([12, 1856])
    Epoch: 1, loss: 0.66147, Train accuracy: 0.61392, Test accuracy: 0.63648
    Epoch: 2, loss: 0.55146, Train accuracy: 0.68837, Test accuracy: 0.70997
    Epoch: 3, loss: 0.47055, Train accuracy: 0.73391, Test accuracy: 0.73885
    Epoch: 4, loss: 0.35673, Train accuracy: 0.77230, Test accuracy: 0.76509
    Epoch: 5, loss: 0.28001, Train accuracy: 0.79521, Test accuracy: 0.77822
    Epoch: 6, loss: 0.35333, Train accuracy: 0.81711, Test accuracy: 0.78346
    Epoch: 7, loss: 0.22455, Train accuracy: 0.83389, Test accuracy: 0.78215
    Epoch: 8, loss: 0.24332, Train accuracy: 0.84761, Test accuracy: 0.77953
    Epoch: 9, loss: 0.29812, Train accuracy: 0.86177, Test accuracy: 0.78215
    Epoch: 10, loss: 0.11444, Train accuracy: 0.87272, Test accuracy: 0.79003
    ```

    It's not entirely train_test_split luck because randomizing split_random_state
    still gets OK results:

    ```
    python train.py \
        --conv_output_size=32 \
        --groups=1 \
        --embedding_size=50 \
        --in_channels=32 \
        --epochs=20 \
        --numpy_random_state=433 \
        --torch_random_state=433994

    INFO:__main__:DEFAULT: split_random_state: 1461995
    WARNING:__main__:NEW KWARGS: numpy_random_state: 433 (<class 'int'>)
    WARNING:__main__:NEW KWARGS: torch_random_state: 433994 (<class 'int'>)

    Epoch: 1, loss: 0.80616, Train accuracy: 0.58999, Test accuracy: 0.68635
    Epoch: 2, loss: 0.79108, Train accuracy: 0.67465, Test accuracy: 0.68504
    Epoch: 3, loss: 0.53284, Train accuracy: 0.72530, Test accuracy: 0.69816
    Epoch: 4, loss: 0.55834, Train accuracy: 0.76368, Test accuracy: 0.70341
    Epoch: 5, loss: 0.58383, Train accuracy: 0.79054, Test accuracy: 0.71654
    Epoch: 6, loss: 0.53005, Train accuracy: 0.80572, Test accuracy: 0.71785
    Epoch: 7, loss: 0.53331, Train accuracy: 0.82514, Test accuracy: 0.72047
    Epoch: 8, loss: 0.36223, Train accuracy: 0.84688, Test accuracy: 0.73360
    Epoch: 9, loss: 0.23334, Train accuracy: 0.85345, Test accuracy: 0.75984
    Epoch: 10, loss: 0.37311, Train accuracy: 0.86878, Test accuracy: 0.75984
    Epoch: 11, loss: 0.16161, Train accuracy: 0.88046, Test accuracy: 0.77034
    Epoch: 12, loss: 0.17307, Train accuracy: 0.89155, Test accuracy: 0.76378
    Epoch: 13, loss: 0.24728, Train accuracy: 0.90220, Test accuracy: 0.76509
    Epoch: 14, loss: 0.24151, Train accuracy: 0.90804, Test accuracy: 0.75853
    Epoch: 15, loss: 0.45944, Train accuracy: 0.91680, Test accuracy: 0.76378
    Epoch: 16, loss: 0.06032, Train accuracy: 0.92527, Test accuracy: 0.76509
    Epoch: 17, loss: 0.16179, Train accuracy: 0.92746, Test accuracy: 0.76640
    Epoch: 18, loss: 0.13890, Train accuracy: 0.93359, Test accuracy: 0.76115
    Epoch: 19, loss: 0.15811, Train accuracy: 0.93913, Test accuracy: 0.75066
    Epoch: 20, loss: 0.05061, Train accuracy: 0.93665, Test accuracy: 0.76115
    """
    # pipeline_kwargs = dict(HYPERPARAMS.parse_args())
    pipeline_args, pipeline_kwargs = utils.parse_argv(sys.argv)

    if len(pipeline_args):
        log.error(f'main.py does not accept positional args: {pipeline_args}')
    log.warning(f'kwargs: {pipeline_kwargs}')

    results = main(**pipeline_kwargs)
    print("=" * 100)
    print("=========== HYPERPARMS =============")
    print(results['hyperparams'].keys())
    print("=" * 100)
    EXPDIR = Path('experiments')
    EXPDIR.mkdir(exist_ok=True)
    try:
        with (EXPDIR / 'train79.hyperparams.json').open('wt') as f:
            json.dump(results['hyperparams'], f)
    except Exception:
        print('Failed to save hyperparams.json')
    try:
        with (EXPDIR / 'train79.results.joblib').open('wb') as f:
            joblib.dump(results, f)
    except Exception:
        print('Failed to save results.joblib')
    try:
        with (EXPDIR / 'train79.pipeline.joblib').open('wb') as f:
            joblib.dump(results['pipeline'], f)
    except Exception:
        print('Failed to save pipeline.joblib')
