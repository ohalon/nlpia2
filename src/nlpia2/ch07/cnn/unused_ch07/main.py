""" Closes to original implementation
FIXME: Verify predict and compute_accuracy() functions by comparing to older versions in git

$ python main.py
Epoch: 1, loss: 0.71129, Train accuracy: 0.56970, Test accuracy: 0.64698
...
Epoch: 10, loss: 0.38202, Train accuracy: 0.80324, Test accuracy: 0.75984
"""
import time
from collections import Counter
import json
from itertools import chain
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from model import CNNTextClassifier
from nlpia2.language_model import nlp


T0 = 1652404117  # number of seconds since 1970-01-01 as of May 12, 2022
MAX_SEED = 2**32 - 1
DATA_DIR = Path(__file__).parent / 'data'

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
log.setLevel(level=logging.INFO)


def tokenize_spacy(doc):
    return [tok.text for tok in nlp(doc) if tok.text.strip()]


def tokenize_re(doc):
    return [tok for tok in re.findall(r'\w+', doc)]


class Parameters:

    def __init__(self):
        self.seq_len: int = 32
        self.filepath: Path = Path('disaster-tweets.csv')
        self.usecols: tuple = ('text', 'target')
        self.tokenizer: str = 'tokenize_re'

        self.vocab_size: int = 2000

        self.embedding_size: int = 64
        self.kernel_lengths: list = [2, 3, 4, 5]
        self.strides: list = [2, 2, 2, 2]
        self.conv_output_size: int = 32

        self.epochs: int = 10
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


HYPERPARAMS = Parameters()


def pad(sequence, pad_value=0, seq_len=HYPERPARAMS.seq_len):
    log.debug(f'BEFORE PADDING: {sequence}')
    padded = list(sequence)[:seq_len]
    padded = padded + [pad_value] * (seq_len - len(padded))
    log.debug(f'AFTER PADDING: {sequence}')
    return padded


def update_params(params=HYPERPARAMS, **kwargs):
    if kwargs.pop('win', False):
        kwargs['split_random_state'] = 850753
        kwargs['numpy_random_state'] = 704
        kwargs['torch_random_state'] = 704463
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


def load_dataset(params, **kwargs):
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
    vocab = [x[0] for x in counts.most_common(params.vocab_size + params.num_stopwords)]
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
    retval = dict(zip(
        'x_train x_test y_train y_test'.split(),
        train_test_split(
            padded_sequences,
            targets,
            test_size=HYPERPARAMS.test_size,
            random_state=split_random_state)))
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
        super().__init__()
        log.info(kwargs)
        params = update_params(params=self)
        self.__dict__.update(params.__dict__)
        print(vars(self))

        dataset = load_dataset(params, **kwargs)
        self.x_train = dataset['x_train']
        self.y_train = dataset['y_train']
        self.x_test = dataset['x_test']
        self.y_test = dataset['y_test']
        self.model = CNNTextClassifier(params=params, **params.__dict__)

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
        return self

    def predict(self, X=None):

        self.model.eval()  # evaluation mode
        predictions = []

        if X is not None:
            X_batches = zip([X], [[None] * len(X)])
        else:
            X_batches = list(zip(*self.loader_test))[0]
            y_batches = list(zip(*self.loader_test))[1]
        with torch.no_grad():
            for x_batch, y_batch in zip(X_batches, y_batches):
                y_pred = self.model(x_batch).detach().numpy()
                predictions += list(y_pred)
        return predictions

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


def parse_argv(sys_argv=sys.argv):
    argv = list(reversed(sys_argv[1:]))

    pipeline_args = []
    pipeline_kwargs = {}  # dict(tokenizer='tokenize_re')
    while len(argv):
        a = argv.pop()
        if a.startswith('--'):
            if '=' in a:
                k, v = a.split('=')
                k = k.lstrip('-')
            else:
                k = a.lstrip('-')
                v = argv.pop()
            pipeline_kwargs[k] = v
        else:
            pipeline_args.append(a)

    return pipeline_args, pipeline_kwargs


def main():

    pipeline_args, pipeline_kwargs = parse_argv(sys.argv)

    if len(pipeline_args):
        log.error(f'main.py does not accept positional args: {pipeline_args}')
    log.warning(f'kwargs: {pipeline_kwargs}')

    pipeline = Pipeline(**pipeline_kwargs)

    pipeline = pipeline.train()
    hyperparms = pipeline.dump()
    print("=" * 100)
    print("=========== HYPERPARMS =============")
    print(hyperparms)
    print("=" * 100)

    # predictions = pipeline.predict()

    return dict(pipeline=pipeline)


if __name__ == '__main__':
    results = main()
