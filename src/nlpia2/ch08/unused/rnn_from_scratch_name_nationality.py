""" Single-layer RNN "from scratch" in PyTorch

References:
  - https://www.twitch.tv/videos/1498823877"
  - https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
  - https://gitlab.com/tangibleai/nlpia2/-/blob/main/src/nlpia2/ch08/rnn_from_scratch_name_nationality.py

Future work:
  - named entity recognizer for misspelled words/typos
  - named entity recognizer for drug names in any language (multilingual)
  - classify first names or full names for nationality
  - classify company names as nonprofits, for profits
  - regressor to estimate business size based on their name only
  - named entity recognizer to identify

Exercises suggested by Shawn Robertson:
  - Try with a different dataset of line -> category, for example:
    - Any word -> language
    - First name -> gender
    - Character name -> writer
    - Page title -> blog or subreddit
  - Get better results with a bigger and/or better shaped network
    - Add more linear layers
    - Try the nn.LSTM and nn.GRU layers
    - Combine multiple of these RNNs as a higher level network

"""
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import pandas as pd
from nlpia2.init import SRC_DATA_DIR, maybe_download
import seaborn as sns


from nlpia2.string_normalizers import Asciifier, ASCII_NAME_CHARS

name_char_vocab_size = len(ASCII_NAME_CHARS) + 1  # Plus EOS marker

# Transcode Unicode str ASCII without embelishments, diacritics (https://stackoverflow.com/a/518232/2809427)
asciify = Asciifier(include=ASCII_NAME_CHARS)


def find_files(path, pattern):
    return Path(path).glob(pattern)


# !curl -O https://download.pytorch.org/tutorial/data.zip; unzip data.zip


labeled_lines = []
categories = []
for filepath in find_files(SRC_DATA_DIR / 'names', '*.txt'):
    filename = Path(filepath).name
    filepath = maybe_download(filename=Path('names') / filename)
    with filepath.open() as fin:
        lines = [asciify(line.rstrip()) for line in fin]
    category = Path(filename).with_suffix('')
    categories.append(category)
    labeled_lines += list(zip(lines, [category] * len(lines)))

df = pd.DataFrame(labeled_lines, columns=('name', 'category'))
n_categories = len(categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
                       'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                       'the current directory.')

print(f'{n_categories} categories:\n{categories}')
print(f'asciify("O’Néàl") => {asciify("O’Néàl")}')


# Find letter index from all_letters, e.g. "a" = 0


all_chars = i2char = list(set(chain(*list(df['name']))))
char2i = dict(zip(i2char, range(len(i2char))))
vocab_size = len(i2char)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor


def one_hot_tensor(c, char2i=char2i):
    """ One-hot encoding of a single character

    >>> one_hot_tensor("A")
    """
    tensor = torch.zeros(1, len(char2i))
    tensor[0][char2i[c]] = 1
    return tensor


def one_hot_sequence(line, char2i=char2i):
    """ One-hot sequence encoding of a line (str) with one row vector for each character

    >>> one_hot_sequence("Abba")
    """
    tensor = torch.zeros(len(line), 1, len(char2i))
    for i, c in enumerate(line):
        tensor[i][0][char2i[c]] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=128,
                 output_size=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, letter_vec, hidden):
        """ Only do forward inference for one time step """
        print(f'letter_vec: {letter_vec.shape}\nhidden: {hidden.shape}')
        cat_input_hidden = torch.cat((letter_vec, hidden), 1)
        hidden = self.i2h(cat_input_hidden)
        output = self.i2o(cat_input_hidden)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def evaluate(self, line_tensor, hidden):
        for i in range(line_tensor.size()[0]):
            output, hidden = self.__call__(line_tensor[i], hidden)

        return output

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, self.hidden_size)


rnn = RNN(vocab_size, hidden_size=128, output_size=n_categories)

next_output, next_hidden = rnn(
    one_hot_sequence('A', char2i=char2i),
    hidden=torch.zeros(1, 1, rnn.hidden_size))


def output2category(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i


print(output2category(next_output))

groups = df.groupby('categories')


def sample():
    """ Retrieve a single example with equal probability form all categories """
    category = random.choice(categories)
    line = groups[category].sample(1)['line']
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = one_hot_sequence(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = sample()
    print('category =', category, '/ line =', line)

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
criterion = nn.NLLLoss()


def train_example(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()


NUM_SAMPLES = 10000
for it in range(1, NUM_SAMPLES + 1):
    category, line, category_tensor, line_tensor = sample()
    output, loss = train_example(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if not it % 100:
        guess, guess_i = output2category(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (it, it / NUM_SAMPLES * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if not it % 100 == 0:
        all_losses.append(current_loss / 100)
        current_loss = 0


plt.figure()
sns.set_theme('notebook')
sns.set_style()

plt.plot(all_losses, grid='on')
plt.show(block=False)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = sample()
    output = rnn.evaluate(line_tensor)
    guess, guess_i = output2category(output)
    category_i = categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + categories, rotation=90)
ax.set_yticklabels([''] + categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
