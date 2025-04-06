from pathlib import Path
import torch
import gzip

EOS_TOKEN = '<eos>'
DEVICE = device = torch.device('cpu')  # 'cuda', 'cuda:0', 'cuda:1'


def smartopen(filepath):
    """ Open file in 'rb' mode (binary stream) gzipped, ascii, or unicode text files. """
    stream = Path(filepath).open('rb')
    if stream.name.lower().endswith('.gz'):
        return gzip.open(stream)
    return stream


class Vocabulary:

    def __init__(self, filepath=None, tokenize_fun=str.split, eos_tok=EOS_TOKEN):
        self.eos_tok = eos_tok
        self.word2idx = {}
        self.idx2word = []
        self.tokenize = tokenize_fun
        if filepath is not None:
            self.add_vocab_from_file(filepath)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_vocab_from_file(self, filepath):
        with smartopen(filepath) as f:
            for line in f:
                # skip blank lines
                if not line.strip():
                    continue
                words = self.tokenize(line.decode()) + [self.eos_tok]
                for word in words:
                    self.add_word(word)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, datadir, corpus_size=10000, raw=False):
        datadir = Path(datadir)
        if not datadir.is_dir():
            datadir = Path(__file__).parent / datadir

        self.vocab = Vocabulary()
        self.vocab.add_vocab_from_file(datadir / 'train.txt')

        # avoid OOV complication by including train+test+validation tokens in vocab
        self.vocab.add_vocab_from_file(datadir / 'test.txt')
        self.vocab.add_vocab_from_file(datadir / 'valid.txt')

        # use the vocab to convert texts into sequences of token IDs
        self.train = self.tokens2ids(datadir / 'train.txt')
        self.test = self.tokens2ids(datadir / 'test.txt')
        self.valid = self.tokens2ids(datadir / 'valid.txt')

    def tokens2ids(self, filepath):
        filepath = Path(filepath)
        assert filepath.is_file()

        with filepath.open() as fin:
            idss = []
            for line in fin:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.vocab.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids  # id_sequences (1-D tensors of id numbers)


def batchify(dataset, batch_size=20, device=DEVICE):
    """ Arrange a dataset (sequence of token ids) into columns for more efficient GPU processing

    For instance, with the alphabet ('a b c ... w x') as the input dataset,
    and with `batch_size=4`, you'd get:

    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘

    shape = (seq_len, batch_size)

    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    """

    # Even number of segments or batches of text
    num_segments = dataset.size(0) // batch_size
    # Trim off any extra text that wouldn't cleanly fit in a batch
    dataset = dataset.narrow(0, 0, num_segments * batch_size)
    # Evenly divide the data across the bsz batches.
    dataset = dataset.view(batch_size, -1).t().contiguous()
    return dataset.to(device)


def batchify_slow(x, batch_size=8, num_batches=5):
    """ Same as batchify, but only creates a single column of data (for slower, more accurate training)."""
    batches = []
    for i in range(int(len(x) / batch_size)):
        if i >= num_batches:
            break
        batches.append(x[i * batch_size:i * batch_size + batch_size])
    return torch.stack(batches)
