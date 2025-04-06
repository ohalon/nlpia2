from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ch08_rnn_char_nationality import RNN, train, save_results, CATEGORIES, CHAR2I, META


class RNNTagger(RNN):

    def __init__(self, n_hidden=128, categories=CATEGORIES, char2i=CHAR2I):
        super().__init__()
        self.categories = categories
        self.n_categories = len(self.categories)  # <1> n_categories = n_outputs (one-hot)
        print(f'RNN.categories: {self.categories}')
        print(f'RNN.n_categories: {self.n_categories}')

        self.char2i = dict(char2i)
        self.vocab_size = len(self.char2i)

        self.n_hidden = n_hidden

        self.W_c2h = nn.Linear(self.vocab_size + self.n_hidden, self.n_hidden)
        self.W_c2y = nn.Linear(self.vocab_size + self.n_hidden, self.n_categories)
        self.activation = nn.Sigmoid()

    # .Tagging in PyTorch
    # [source,python]
    # ----
    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.W_c2h(combined)
        y = self.W_c2y(combined)
        y = self.activation(y)  # <1>
        return y, hidden
    # ----
    # <1> if you use BCEWithLogitsLoss you can delete the activation layer for faster training

    def __str__(self):
        return (
            f"RNNTagger(\n    n_hidden={self.n_hidden},\n    n_categories={self.n_categories},\n"
            f"    categories=[{self.categories[0]}..{self.categories[-1]}],\n"
            f"    vocab_size={self.vocab_size},\n    char2i['A']={self.char2i['A']}\n)"
        )


def multihot_encoder(labels, dtype=torch.float32):
    """ Convert an array of label lists into a 2-D multihot tensor (array of vectors)

    label_lists = [['happy', 'kind'], ['sad', 'mean'], ['loud', 'happy'], ['quiet', 'kind']]
    >>> multihot_encoder(label_lists, dtype=None)
    """
    label_set = set()
    for label_list in labels:
        label_set = label_set.union(set(label_list))
    label_set = sorted(label_set)
    multihot_vectors = []
    # If you want to keep track of your labels and where they are in your vectors:
    # label2id = {v: k for k, v in enumerate(label_set)}
    for label_list in labels:
        multihot_vectors.append([1 if x in label_list else 0 for x in label_set])
    # # You probably want to keep track of which columns are which so you should store your data in a DataFrame/csv
    # # before training a model on it
    if dtype is None:
        return pd.DataFrame(multihot_vectors, columns=label_set)
    return torch.Tensor(multihot_vectors).to(dtype)


def create_multihot_dataset(df, normalize=True, fillna=0, text_col='surname', target_col='nationality'):
    name_multihot_vecs = {}
    # FIXME: this dataset has already been deduplicated,
    #        so use the 'count' column instead of counting the nationality labels
    for text, group in df.groupby(text_col):
        name_multihot_vecs[text] = Counter(group[target_col])
    tags = pd.DataFrame(name_multihot_vecs).T.fillna(0)
    tags2 = pd.DataFrame()
    sums = tags.T.sum()
    for c in tags.columns:
        tags2[c] = tags[c] / sums
    return tags2


if __name__ == '__main__':
    repo = 'tangibleai/nlpia2'
    filepath = 'src/nlpia2/data/surname_nationalities.csv'
    suffix = '?inline=false'
    url = f"https://gitlab.com/{repo}/-/raw/main/{filepath}{suffix}"
    df = pd.read_csv(url)
    print(df)

    n_categories = 10
    ans = input(f"How many nationalities would you like to train on? [{n_categories}]? ")
    if ans.strip():
        n_categories = int(ans)
    categories = sorted(df['nationality'].unique())[:n_categories]
    print(f"categories: {categories}")

    char2i = META['char2i']
    char2i = dict(zip(sorted(char2i), range(len(char2i))))
    n_hidden = 128
    model = RNNTagger(
        char2i=char2i,
        categories=categories,
        n_hidden=128
    )
    print(f"model: {model}")

    n_iters = 10000
    ans = input(f"How many samples would you like to train on? [{n_iters}]? ")
    if ans.strip():
        n_iters = int(ans)

    lr = .005
    ans = input(f"What learning rate would you like to train with? [{lr}]? ")
    if ans.strip():
        lr = float(ans)

    criterion = nn.BCELoss()

    results = dict(lr=lr, n_iters=n_iters, criterion=criterion)
    print(f"hyperparams: {results}")

    df_multihot = create_multihot_dataset(df)
    categories = list(df_multihot.columns)
    df_multihot['nationality'] = tuple(df_multihot.values)
    df_multihot['surname'] = df_multihot.index.values
    df_multihot = df_multihot['surname nationality'.split()].reset_index(drop=True)

    # user can abort by setting any hyperparamter to 0 or False or None
    if n_iters and n_hidden and lr:
        training_results = train(model=model, df=df_multihot, n_iters=n_iters, criterion=criterion, lr=lr)
        results.update(training_results)
        print(f"updated results: {results}")

        # required for computing the filename
        results['train_time'] = results.get('train_time', f'{np.random.randint(1000)}:np.random.randint(100)')
        results['losses'] = results.get('losses', [99])

        save_results(**results)
