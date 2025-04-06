"""
PyTorch translation encoder-decoder tutorial (autoencoder exercise)

References:
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://opendata.stackexchange.com/q/3888/17196
* https://tatoeba.org/eng/downloads
* https://gitlab.com/tangibleai/nlpia2/
* https://www.manythings.org/anki/
"""
from __future__ import unicode_literals, print_function, division
import unicodedata
import re
import random
import time
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np
import seaborn as sns  # noqa
from urllib.request import urlretrieve  # , urljoin

# from nlpia2.netutils import download_if_necessary
# from nlpia2.init import maybe_download
# from nlpia2.constants import SRC_DATA_DIR as DATA_DIR
log = logging.getLogger(__name__)

try:
    DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
except Exception:
    DATA_DIR = Path.cwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if reverse=True then English is LANG2 else English is LANG1
LANG1, LANG2 = 'eng', 'ukr'
SOS_token = 0
EOS_token = 1

HIDDEN_SIZE = 128
NUM_EPOCHS = 80
BATCH_SIZE = 32


def download_if_not_exists(
        url,
        dest_filepath,
        force=False,
        data_dir=DATA_DIR):
    """ Download a file only if it has not yet been cached locally in ~/.qary-data/ HOME_DATA_DIR
    """
    dest_filepath = Path(dest_filepath)
    if dest_filepath.is_file():
        return dest_filepath
    if len(dest_filepath.parts) == 1:
        dest_filepath = DATA_DIR / dest_filepath

    if not dest_filepath.parent.is_dir():
        dest_filepath.parent.mkdir(parents=True, exist_ok=True)  # FIXME add , reporthook=DownloadProgressBar())

    if force or not dest_filepath.is_file():
        log.warning(f"Downloading: {url} to {dest_filepath}")
        dest_filepath, _ = urlretrieve(str(url), str(dest_filepath))
        log.warning(f"Finished downloading '{dest_filepath}'")
        return dest_filepath


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def to_ascii(s):
    """Turn Unicode str to plain ASCII with unicodedata.normalize('NFD', str)

    SEE: https://stackoverflow.com/a/518232/2809427
    """

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalize_str(s, ascii_only=False, lower=True, ignore=r"'", punctuation=r"[:;,.!?]"):
    """ convert unicode characters to their ascii equivalent, such as direcitonal apostrophes """
    # FIXME: even with normalize_unicode off, need to deal with ?, ', " and their unicode equivalent (see qary)
    s = s.strip()
    if lower:
        s = s.lower()
    if ascii_only:
        s = to_ascii(s)
    s = re.sub(f"({punctuation})", r" \1", s)  # spaces before punctuation
    if ignore:
        s = re.sub(ignore, r" ", s)
    if ascii_only:
        s = re.sub(r"[^a-zA-Z!?.]+", r" ", s)  # removes sentence-terminating period
    return s.strip()


def read_langs(lang1=LANG1, lang2=LANG2, reverse=False):
    """ Read pairs of translation str from Anki-style tsv files to create list of 2-lists

    Only the eng-ukr and eng-spa Anki files have been downloaded to the nlpia2 DATA_DIR
    For other languages you must manually download, unzip, and rename the Anki source files:
    * https://www.manythings.org/anki/

    Anki stores translation pairs in a 3-column tab-separated file at {DATA_DIR}/{lang2}.tsv.
    The third column is an unused column containing "LICENSE" and attribution info.

    Args:
      lang1 (str): 3-letter ISO language code of the source text (first column)
      lang2 (str): 3-letter ISO language code of the target language (second column)
      reverse (bool): Whether to swap the source and target text strs

    Returns:
      input_lang (str): 3-letter ISO language code of the source text (first column)
      output_lang (str): 3-letter ISO language code of the target language (second column)
      pairs (list([str, str])): 2-D array of strings, column 1 = source text, column 2 = target text
    """
    url = f'https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2/data/{lang2}.tsv?inline=false'
    dest_path = (DATA_DIR / lang2).with_suffix('.tsv')
    print(url, dest_path)
    download_if_not_exists(url, dest_filepath=dest_path)
    df = pd.read_csv(url, sep='\t')
    df.columns = f'{lang1} {lang2} license'.split()

    desc = df.describe(include='all')
    print(desc)
    pairs = [
        [normalize_str(s1), normalize_str(s2)]
        for s1, s2 in zip(list(df[lang1]), list(df[lang2]))
    ]
    df = pd.DataFrame(pairs)
    desc = df.describe(include='all')
    print(desc)

    if reverse:
        pairs = [[p[1], p[0]] for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 16

# unused
CONTRACTIONS = {
    "don't": "do not", "isn't": "is not", "can't": "cannot",
    "we're": "we are", "i'm": "i am", "how'd": "how did", "how're": "how are",
    "what's": "what is", "he's": "he is", "she's": "she is", "where's": "where is",
    "they're": "they are", "how's": "how is", "who's": "who is", "who're": "who are",
}

# fewer prefixes will reduce training data and time and increase overfitting
ENG_PREFIXES = (
    "are ", "is ", "am ", "do ", "does ", "can ", "may ",
    "i ", "you ", "my ", "he ", "she ", "they ", "it ", "we ",
    "how ", "who ", "what ", "when ", "where ", "why ",

    "how are ", "how re ", "how is ", "how s", "how do", "how d ",
    "i am ", "i m ", "he is", "he s ", "she is ", "she s ",
    "you are ", "you re ", "we are ", "we re ", "they are ", "they re "
)


def filter_pair(p, reverse=False, eng_prefixes=ENG_PREFIXES):
    """ Add spaces to punctuation and tokenize all pairs with str.split()
    FIXME: use SpaCy language model to tokenize everything and skip `normalize_str`
    """
    return (
        len(p[0].split(' ')) < MAX_LENGTH
        and len(p[1].split(' ')) < MAX_LENGTH
        and (not eng_prefixes or p[int(reverse)].startswith(eng_prefixes))
    )


def filter_pairs(pairs, reverse=False):
    return [pair for pair in pairs if filter_pair(pair, reverse=reverse)]


def prepare_data(lang1=LANG1, lang2=LANG2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse=reverse)
    print(input_lang.name, output_lang.name, random.choice(pairs))
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs, reverse=reverse)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print(input_lang.name, output_lang.name, random.choice(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(lang1=LANG1, lang2=LANG2, batch_size=BATCH_SIZE):
    input_lang, output_lang, pairs = prepare_data(lang1=LANG1, lang2=LANG2, )

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def plot(training_log):
    df = pd.DataFrame(training_log)
    df.plot(df['epoch'], df['loss'], logy=True)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          learning_rate_alpha=.9999, print_every=100):
    start = time.time()
    training_log = []
    print_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print('running training...')
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        training_log.append(
            dict(loss=loss, epoch=epoch, learning_rate=learning_rate,
                 time=time.time() - start))
        learning_rate *= learning_rate_alpha

    return dict(encoder=encoder, decoder=decoder, training_log=training_log)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluate_randomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    hidden_size = HIDDEN_SIZE
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    lang1 = LANG1
    lang2 = LANG2

    # FIXME: argsdict = argparse.parseargs()

    input_lang, output_lang, train_dataloader = get_dataloader(
        lang1=lang1, lang2=lang2, batch_size=batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    results = train(train_dataloader, encoder, decoder,
                    num_epochs, print_every=5)
    i = 0
    path = DATA_DIR / f'{lang1}-{lang2}.{i}.joblib'
    while path.is_file():
        i += 1
        path = DATA_DIR / f'{lang1}-{lang2}.{i}.joblib'
    joblib.dump(results, path)
    df = pd.DataFrame(results['training_log']).set_index('epoch')
    df.plot(df['epoch'], df['loss'], logy=True)
