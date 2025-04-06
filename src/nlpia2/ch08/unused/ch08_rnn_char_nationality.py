# -*- coding: utf-8 -*-
"""

    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish

    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch
"""
from collections import Counter
import copy
from pathlib import Path
import time

import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from nlpia2.init import SRC_DATA_DIR, maybe_download
from nlpia2.string_normalizers import Asciifier, ASCII_NAME_CHARS

from persistence import save_model

MODEL_PATH = Path(__file__).with_suffix('').name
PYTORCH_TUTORIAL_CATEGORIES = [
    'Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese',
    'Korean', 'Nigerian', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese'
]
MANUALLY_ADDED_CATEGORIES = ['Ethiopian', 'Indian', 'Nepalese']


# META = load_model_meta(MODEL_PATH)

META = {
    'categories': [
        'Algerian', 'Arabic', 'Brazilian', 'Chilean', 'Chinese', 'Czech', 'Dutch', 'English', 'Ethiopian',
        'Finnish', 'French', 'German', 'Greek', 'Honduran', 'Indian', 'Irish', 'Italian', 'Japanese', 'Korean',
        'Malaysian', 'Mexican', 'Moroccan', 'Nepalese', 'Nicaraguan', 'Nigerian', 'Palestinian', 'Papua New Guinean',
        'Peruvian', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'South African', 'Spanish', 'Ukrainian',
        'Venezuelan', 'Vietnamese'
    ],
    'char2i': {
        ' ': 0, "'": 1, ',': 2, '-': 3, '.': 4, ';': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10,
        'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21,
        'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, 'a': 32, 'b': 33,
        'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44, 'n': 45,
        'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55, 'y': 56, 'z': 57
    },
}
META['n_hidden'] = 128
META['n_categories'] = len(META['categories'])
# save_model(MODEL_PATH, **META)

CATEGORIES = META['categories']
CHAR2I = META['char2i']


class RNN(nn.Module):

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
        self.softmax = nn.LogSoftmax(dim=1)

    # .Recurrence in PyTorch
    # [source,python]
    # ----
    def forward(self, x, hidden):  # <1>
        combined = torch.cat((x, hidden), 1)  # <2>
        hidden = self.W_c2h(combined)  # <3>
        y = self.W_c2y(combined)  # <4>
        y = self.softmax(y)
        return y, hidden    # <5>
    # ----
    # <1> token (character) one-hot vector
    # <2> concatenate the `x` vector and the previous character's `hidden` tensor
    # <3> `nn.Linear` dot product transforms `combined` vector into a `hidden` vector
    # <4> dot product transforms `combined` vector into `y` (output vector of category likelihoods)
    # <5> Both `output` and `hidden` tensors are needed to process the next token

    def encode_one_hot_vec(self, character):
        """ one - hot encode a single char """
        tensor = torch.zeros(1, len(self.char2i))
        tensor[0][self.char2i[character]] = 1
        return tensor

    def encode_one_hot_seq(self, text):
        """ one - hot encode each char in a str = > matrix of size(len(str), len(alphabet)) """
        tensor = torch.zeros(len(text), 1, len(ASCII_NAME_CHARS))
        for pos, letter in enumerate(text):
            tensor[pos][0][self.char2i[letter]] = 1
        return tensor

    def evaluate_tensor(self, text_tensor):
        with torch.no_grad():
            hidden = torch.zeros(1, self.n_hidden)
            for i in range(text_tensor.shape[0]):
                output, hidden = self(text_tensor[i], hidden)
        return output

    def category_from_output(self, output_tensor):
        top_n, top_i = output_tensor.topk(1)
        category_i = top_i[0].item()
        return self.categories[category_i], category_i

    def predict_category(self, text):
        pred_i = self.predict_category_index(text)
        return self.categories[pred_i]

    def predict_category_index(self, text):
        tensor = self.encode_one_hot_seq(text)
        return self.evaluate_tensor(tensor).topk(1)[1][0].item()

    def predict_category_and_index(self, text):
        tensor = self.encode_one_hot_seq(text)
        pred_i = self.evaluate_tensor(tensor).topk(1)[1][0].item()
        return self.categories[pred_i], pred_i

    def load(self, filepath):
        filepath = Path(filepath)
        with filepath.with_suffix('.meta.json').open('rt') as fin:
            self.meta = json.load(fin)
        self.__init__(
            n_hidden=self.meta['n_hidden'],
            char2i=self.meta.get('char2i', None),
            categories=self.meta.get('categories', None))
        with filepath.with_suffix('.state_dict.pickle').open('rb') as fin:
            state_dict = torch.load(fin)
        self.load_state_dict(state_dict)
        return self

    def save(self, filepath):
        """ Save met to filepath.meta.json & state_dict to filepath.state_dict.pickle """
        filepath = Path(filepath)
        filedir = filepath.parent
        meta = dict(
            filedir=str(filedir),
            state_dict_filename=filepath.with_suffix('.state_dict.pickle').name,
            meta_filename=filepath.with_suffix('.meta.json').name,
            n_hidden=self.n_hidden,
            categories=self.categories,
            n_categories=self.n_categories,
            char2i=self.char2i,
            vocab_size=self.vocab_size)
        with (filedir / meta['meta_filename']).open('wt') as fout:
            json.dump(meta, fout, indent=4)
        state_dict = self.state_dict()
        with (filedir / meta['state_dict_filename']).open('wb') as fout:
            torch.save(state_dict, fout)
        return filepath

    def unroll_activations(self, text):
        char_seq_tens = encode_one_hot_seq(text, char2i=self.char2i)
        hidden = torch.zeros(1, self.n_hidden)
        self.zero_grad()
        outputs = []
        hiddens = []
        for i in range(char_seq_tens.size()[0]):
            output, hidden = self(char_tens=char_seq_tens[i], hidden=hidden)
            outputs.append(output)
            hiddens.append(hidden)
        cats = []
        for v in outputs:
            cats.append(self.categories[np.exp(v.detach().numpy()).argmax()])
        return cats, outputs, hiddens

    def __str__(self):
        return (
            f"RNN(\n    n_hidden={self.n_hidden},\n    n_categories={self.n_categories},\n"
            f"    categories=[{self.categories[0]}..{self.categories[-1]}],\n"
            f"    vocab_size={self.vocab_size},\n    char2i['A']={self.char2i['A']}\n)"
        )


asciify = Asciifier(include=ASCII_NAME_CHARS)


def dedupe_mapping_df(df, key_column='surname', value_column='nationality'):
    key_value_tuples = list(zip(df[key_column], [value_column]))
    key_value_counts = [[k[0], k[1], v] for (k, v) in Counter(key_value_tuples).items()]
    return pd.DataFrame(key_value_counts, columns=[key_column, value_column, 'count'])


def dataset_confusion(df, normalize=True, fillna=0, text_col='surname', target='nationality'):
    """ Given a df with columns name & category, assume "truth" is most popular category for a name """
    confusion = {c: Counter() for c in sorted(df[target].unique())}
    for i, g in df.groupby(text_col):
        counts = Counter(g[target])
        confusion[counts.most_common()[0][0]] += counts
    confusion = pd.DataFrame(confusion)
    confusion = confusion[confusion.index]
    if normalize:
        confusion /= confusion.sum(axis=1)
    if fillna is not None:
        confusion.fillna(fillna, inplace=True)
    confusion.index.name = 'most_common'
    return confusion


def encode_one_hot_vec(letter, char2i=CHAR2I):
    """ one-hot encode a single character using the char2i mapping of chars to ints """
    tensor = torch.zeros(1, len(char2i))
    tensor[0][char2i[letter]] = 1
    return tensor


def encode_one_hot_seq(line, char2i=CHAR2I):
    """ one-hot encode each char in a str => matrix of size(len(str), len(alphabet)) """
    tensor = torch.zeros(len(line), 1, len(ASCII_NAME_CHARS))
    for pos, letter in enumerate(line):
        tensor[pos][0][char2i[letter]] = 1
    return tensor


def sample_groupby(df, num_samples=1, groupby='nationality', char2i=CHAR2I, replace=True, shuffle=True):
    """ balanced sampling of all categories """
    if sample_groupby.groups is None:
        sample_groupby.groups = df.groupby(groupby)
    df_sample = sample_groupby.groups.sample(num_samples, replace=replace)
    if shuffle:
        df_sample = df_sample.sample(len(df_sample))
    return df_sample


sample_groupby.groups = None


def random_example(groups, target='nationality', text_col='surname', categories=None, char2i=None):
    """ balanced sampling of all categories """
    # ANTIPATTERN
    # random_example.df = getattr(random_example, 'df', df)
    # if 'count' not in df.columns:
    #     random_exmaple.df = dedupe_mapping_df(df)
    row = groups.sample(1).sample(1)
    name = row[text_col].iloc[0]
    category = row[target].iloc[0]
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = encode_one_hot_seq(name, char2i=char2i)
    return category, name, category_tensor, line_tensor


def stratified_random_examples(
        groups, num_samples_per_group=1, replace=True, shuffle=True,
        target='nationality', text_col='surname', categories=None, char2i=None,
        multihot=False):
    """ balanced sampling of all categories """
    # print(f'groups {groups}')
    # print(f'df = groups.sample({num_samples_per_group}, replace={replace})')
    # print(f'groups.sample(1): {groups.sample(1)}')

    try:
        df = groups.sample(num_samples_per_group, replace=replace)
    except Exception as e:
        print(e)
        df = groups
    if shuffle:
        df = df.sample(len(df))
    names = df[text_col].values
    cats = df[target].values
    tqdm_fun = tqdm if len(cats) > 10000 else iter

    # if the target variable (category) is a str then look it up in categories and create a one-hot vector
    if isinstance(cats[0], str):
        cat_tensors = [
            torch.tensor([categories.index(c)], dtype=torch.long) for c in
            tqdm_fun(cats)
        ]
    # if the target variable (category) is a vector/array then convert it to a multihot float tensor
    else:
        cat_tensors = [
            torch.tensor(c, dtype=torch.float) for c in
            tqdm_fun(cats)
        ]
    line_tensors = [
        encode_one_hot_seq(n, char2i=char2i) for n in tqdm_fun(names)
    ]
    return cats, names, cat_tensors, line_tensors


# TODO: move this into a class called Pipeline or Trainer that inherits RNN or Model
def train_sample(model, category_tensor, char_seq_tens,
                 criterion=None, lr=.005):
    """ Train for one epoch (one example name nationality tensor pair) """
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    for char_onehot_vector in char_seq_tens:
        category_predictions, hidden = model(
            x=char_onehot_vector, hidden=hidden)
#    log internal state to a global variable or file so that it can be visualized
#    print(f"category_predictions: {category_predictions}")
#    print(f"category_tensor: {category_tensor}")
    loss = criterion(category_predictions, category_tensor)
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return model, category_predictions, loss.item()


CRITERION = nn.NLLLoss()


def visualize_outputs(model, text):
    char_seq_tens = encode_one_hot_seq(text, char2i=model.char2i)
    hidden = torch.zeros(1, model.n_hidden)
    model.zero_grad()
    outputs = []
    hiddens = []
    for i in range(char_seq_tens.size()[0]):
        output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
        outputs.append(output)
        hiddens.append(hidden)
    cats = []
    for v in outputs:
        cats.append(model.categories[np.exp(v.detach().numpy()).argmax()])
    return cats


def train_batch(df_batch, model, categories, target='nationality', text_col='surname', criterion=CRITERION, lr=.005, char2i=CHAR2I):
    """ train for one epoch(one batch of example tensors) """
    output_losses = []
    for i, row in df_batch.iterrows():
        category_tensor = torch.tensor([categories.index(row[target])], dtype=torch.long)
        line_tensor = encode_one_hot_seq(row[text_col], char2i=char2i)
        model, output, loss = train_sample(
            model=model,
            category_tensor=category_tensor,
            char_seq_tens=line_tensor,
            criterion=criterion,
            lr=lr)
        output_losses.append((output, loss))
    return model, output_losses


def time_elapsed(t0):
    """ Compute time since t0(t0=time.time() in seconds) """
    secs = time.time() - t0
    mins = secs // 60
    secs = int(secs - mins * 60)
    mins = int(mins)
    return f'{mins:02d}:{secs:02d}'


def confusion_df(truth, pred, categories=None):
    """ Count mislabeled examples in entire dataset """
    pair_counts = Counter(zip(truth, pred))
    confusion = {c_tru: {c_pred: 0 for c_pred in categories} for c_tru in categories}
    for ((t, p), count) in pair_counts.items():
        confusion[t][p] = count
    return pd.DataFrame(confusion)


def predict_confusion(df, categories=None, target='nationality', text_col='surname'):
    df_conf = confusion_df(
        truth=df[target],
        pred=df[text_col].apply(model.predict_category).values,
        categories=categories,
    )
    return df_conf


def plot_confusion(df_conf):
    df_conf = df_conf.replace('', 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_conf.values)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + list(df_conf.columns), rotation=90)
    ax.set_yticklabels([''] + list(df_conf.index))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def topk_predictions(model, text, target_col='nationality', topk=3):
    with torch.no_grad():
        output = model.evaluate_tensor(encode_one_hot_seq(text))
        topvalues, topindices = output.topk(topk, 1, True)
        predictions = []
        # TODO: try this:
        for rank, (log_loss_tens, category_index) in enumerate(zip(topvalues[0], topindices[0])):
            predictions.append(
                [rank, text, log_loss_tens.item(), model.categories[category_index]])
    return pd.DataFrame(predictions, columns='rank text log_loss'.split() + [target_col])


def predict(model, text):
    with torch.no_grad():
        output = model.evaluate_tensor(model.encode_one_hot_seq(text))
        topvalues, topindices = output.topk(1, 1, True)
        (log_loss_tens, category_index) = topindices[0]
        return model.categories[category_index]


def predict_proba(model, text):
    with torch.no_grad():
        output = model.evaluate_tensor(model.encode_one_hot_seq(text))
        topvalues, topindices = output.topk(1, 1, True)
        (log_loss_tens, category_index) = topindices[0]
        return np.exp(log_loss_tens.item())


def print_predictions(model, text, target_col='nationality', n_predictions=3):
    preds_df = topk_predictions(model=model, text=text, target_col=target_col, topk=n_predictions)
    if n_predictions > 1:
        print(preds_df)
    return preds_df


def print_example_tensor(text="O’Néàl", char2i=CHAR2I):

    # Transcode Unicode str ASCII without embellishments, diacritics (https://stackoverflow.com/a/518232/2809427)
    ascii_text = asciify(text)
    print(f'asciify({text}) => {ascii_text}')

    encoded_char = encode_one_hot_vec(ascii_text[0], char2i=char2i)
    print(f"encode_one_hot_vec({ascii_text[0]}): {encoded_char}")
    input_tensor = encode_one_hot_seq(ascii_text, char2i=char2i)
    print(f"input_tensor.size(): {input_tensor.size()}")


def print_dataset_samples(df, num_samples=3, replace=True, target='nationality'):
    print(sample_groupby(df, num_samples=num_samples, groupby=target, replace=replace))


def load_name_counts(filepath=SRC_DATA_DIR / 'names' / 'name_counts.csv.gz'):
    return pd.read_csv(filepath)


def preprocess_surname_nationality_df(df, target_col='nationality', text_col='surname'):
    new_rows = []
    # Some Ukranian names have Russian alternatives e.g. surname='Markevych (Russian: Markevich)'
    # With the Russian invasion of Ukraine it is important to distinguish between the two
    # (the nationalities and the languages of Russia and Ukraine)
    # issus = df['surname'].str.contains(r'(', regex=False)
    # retain only the Ukranian spelling for Ukranian names:
    df[target_col] = df[target_col].apply(lambda x: asciify(x))
    print(df)
    df[text_col] = df[text_col].str.split('(').apply(lambda x: x[0].strip())
    print(df)
    df[text_col] = df[text_col].apply(lambda x: asciify(x))
    print(df)
    df[text_col] = df[text_col].str.strip().str.strip(',')
    print(df)
    ismulti = ~df[text_col].str.match(r"^[- A-Za-z']+$")
    print(f"sum(ismulti): {sum(ismulti)}")
    if sum(ismulti) > 0:
        for i, row in df[ismulti].iterrows():
            base_row = row.to_dict()
            print(base_row)
            for name in row[text_col].split(','):
                name = name.strip().strip(',')
                new_row = copy.copy(base_row)
                new_row.update({text_col: name})
                new_rows.append(new_row)
        new_rows = pd.DataFrame(new_rows)
        print('NEW ROWS')
        print(new_rows)
        df = df.drop(ismulti, axis=0)
        df = pd.concat([df, new_rows])
    return df


def train(model, df, n_iters=5000, print_every=None, target='nationality', text_col='surname',
          criterion=nn.NLLLoss(), lr=.005, val_split=.05, multihot=False):
    isdataset = df[target].isin(model.categories)
    isvalidationset = np.random.rand(len(isdataset)) < val_split  # 10% validation set
    df_train = df[isdataset & ~isvalidationset].copy()
    df_val = df[isdataset & isvalidationset].copy()

    if multihot or not isinstance(criterion, nn.NLLLoss):
        multihot = True
        # surprisingly you can groupby the onehot vectors
        # (all the 679 different combinations of tags when there are 37 nationalities/categories)
        # FYI 37*36 = 1332 > 679 < .1 * len(df) = .1 * 30923 = 3092.3
        # as long as the vectors are tuples
        df_train[target] = df_train[target].apply(tuple)
    groups = df_train.groupby(target)
    # print(f'groups in train: {groups}')

    mean_train_losses = []
    mean_train_accuracies = []
    mean_val_accuracies = []

    print_every = n_iters // 50 if print_every is None else print_every

    start = time.time()

    batch_losses = []
    batch_predictions = []
    batch_accuracies = []

    criterion = nn.NLLLoss() if criterion is None or not multihot else criterion

    for it in tqdm(range(n_iters)):
        cats, lines, category_tensors, line_tensors = stratified_random_examples(
            groups, num_samples_per_group=1, categories=model.categories, char2i=model.char2i, multihot=multihot)

        for cat, line, cat_tensor, line_tensor in zip(cats, lines, category_tensors, line_tensors):
            model, output_tensor, loss = train_sample(
                model=model, category_tensor=cat_tensor, char_seq_tens=line_tensor, criterion=criterion, lr=lr)
            guess, guess_i = model.predict_category_and_index(line)
            batch_predictions.append(guess)
            batch_accuracies.append(guess == cat)
            batch_losses.append(loss)
            if it and not (it % print_every):
                # print(f'    output_tensor: {output_tensor}\tmodel.category_from_output(output_tensor): {model.category_from_output(output_tensor)}')
                correct = '✓' if batch_accuracies[-1] else f'✗ should be {cat} ({model.categories.index(cat)}={cat_tensor[0].item()})'
                print(f'{it:06d} {(it*100) // n_iters}% {time_elapsed(start)} {loss:.4f} {line} => {guess} ({guess_i}) {correct}')
        if it and not (it % print_every):
            mean_train_losses.append(np.mean(batch_losses))
            mean_train_accuracies.append(np.mean(batch_accuracies))
            mean_val_accuracies.append(np.mean([model.predict_category(text=s) == c for (s, c) in zip(df_val[text_col], df_val[target])]))
            print(
                f"  mean_train_loss: {mean_train_losses[-1]}\n"
                f"  mean_train_acc: {mean_train_accuracies[-1]}\n"
                f"  mean_val_acc: {mean_val_accuracies[-1]}")
            batch_losses = []
            batch_predictions = []
            batch_accuracies = []

    train_time = time_elapsed(start)
    return dict(
        model=model,
        n_hidden=model.n_hidden,
        train_losses=mean_train_losses,
        train_accuracies=mean_train_accuracies,
        validation_accuracies=mean_val_accuracies,
        losses=[1.0 - a for a in mean_val_accuracies],
        train_time=train_time,
        categories=model.categories,
        char2i=model.char2i,
        lr=lr,
        n_iters=n_iters,
        df_len=len(df),
    )


def plot_training_curve(model, losses):
    plt.figure()
    plt.plot(losses)
    plt.show(block=False)

    print(f"META['categories']: {META['categories']}")
    print(f'CATEGORIES: {CATEGORIES}')
    print()
    print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
    print_predictions(model, text='Fyodor', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
    print()
    print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
    print_predictions(model, text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Sanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Koyejo', n_predictions=3, categories=CATEGORIES)
    print()
    print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
    print_predictions(model, text='Satoshi', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Nakamoto', n_predictions=3, categories=CATEGORIES)
    print()
    print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
    print_predictions(model, text='Rediet', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Abebe', n_predictions=3, categories=CATEGORIES)
    print()
    print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
    print_predictions(model, text='Silvio', n_predictions=3, categories=CATEGORIES)
    print_predictions(model, text='Micali', n_predictions=3, categories=CATEGORIES)


def save_results(filename=None, **results):
    # load/save test for use on the huggingface spaces server
    meta = copy.deepcopy(results)
    # meta['model'] = results['model']
    # meta['losses'] = results['losses']
    # meta['train_time'] = results['train_time']

    meta['state_dict'] = results['model'].state_dict()
    if filename is None:
        if 'losses' not in meta:
            meta['losses'] = [np.log(np.random.rand())]
        if 'train_time' not in meta:
            meta['train_time'] = '999:99'
        start_min = len(meta['losses']) // 4
        meta['min_loss'] = min(meta['losses'][start_min:])
        print(f"min_loss: {meta['min_loss']}")
        train_time_str = str(results['train_time']).replace(':', 'min_') + 'sec'
        filename = str(MODEL_PATH) + f"-{meta['min_loss']:.3f}-{train_time_str}"
        filename = filename.replace('.', '_')
    save_model(filename, **meta)
    print(f'Model meta.keys(): {meta.keys()}')
    print(f'Saved model state_dict and meta to {filename}.*')


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
    model = RNN(
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

    results = dict(lr=lr, n_iters=n_iters)
    print(f"results: {results}")
    if n_iters and n_hidden and lr:
        training_results = train(model=model, df=df, n_iters=n_iters, lr=lr)
        results.update(training_results)
        print(f"updated results: {results}")

        # required for computing the filename
        results['train_time'] = results.get('train_time', f'{np.random.randint(1000)}:np.random.randint(100)')
        results['losses'] = results.get('losses', [99])

        save_results(**results)
