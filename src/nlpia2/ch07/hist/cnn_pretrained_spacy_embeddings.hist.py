import torch
from torch import nn
def create_embedding_matrix(word_index,embedding_dict,dimension):
  embedding_matrix=np.zeros((len(word_index)+1,dimension))

  for word,index in word_index.items():
    if word in embedding_dict:
      embedding_matrix[index]=embedding_dict[word]
  return embedding_matrix
import nessvec
from nessvec import futil
from nessvec import file_util
from nessvec import utils
from nessvec import constants
dir(nessvec)
help(nessvec)
def create_embedding_matrix(vocab):
    # embedding_matrix = np.zeros((len(word_index)+1,dimension))
    embeddings = []
    for word in vocab:
        embeddings.append(nlp(word).vector)
    return np.array(embeddings)
import spacy
nlp = spacy.load('en_core_web_md')
def create_embedding_matrix(vocab):
    # embedding_matrix = np.zeros((len(word_index)+1,dimension))
    embeddings = []
    for word in vocab:
        embeddings.append(nlp(word).vector)
    return np.array(embeddings)
pwd
cd src
cd nlpia2
cd ch07
cd cnn
ls -hal
import main
from main import *
who
params = load_dataset()
params = Parameters()
params = load_dataset(params=params)
params = load_dataset(params=params, tokenizer='tokenize_spacy')
param_dict = load_dataset(params=Parameters(), tokenizer='tokenize_spacy')
param_dict
params
params.__dict__
params = Parameters()


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
        padded_sequences.append(pad(s, pad_value=0))
    padded_sequences = torch.IntTensor(padded_sequences)

    # 10. Configurable sampling for testset (test_size samples)
    retval = dict(zip(
        'x_train x_test y_train y_test'.split(),
        train_test_split(
            padded_sequences,
            targets,
            test_size=HYPERPARAMS.test_size,
            random_state=0)))
    retval['vocab'] = vocab
    retval['tok2id'] = tok2id
    return retval
params = Parameters()
param_dict = load_dataset(params=Parameters(), tokenizer='tokenize_spacy')
param_dic t
param_dict
import json
json.dump(param_dict, open("cnn_train_test_data_vocab_id2tok.json", 'wt'), indent=4)
data = {k: list(v) for (k, v) in param_dict.items() if isinstance(v, torch.Tensor)}
param_dict.update(data)
json.dump(param_dict, open("cnn_train_test_data_vocab_id2tok.json", 'wt'), indent=4)
data = {k: list(v.to_numpy()) for (k, v) in param_dict.items() if isinstance(v, torch.Tensor)}
data
data = {k: list(v.to_numpy()) for (k, v) in param_dict.items()}
data = {k: list(v) for (k, v) in param_dict.items()}
data
del data['tok2id']
json.dump(data, open("cnn_train_test_data_vocab_id2tok.json", 'wt'), indent=4)
hist -f /home/hobs/code/tangibleai/nlpia2/src/nlpia2/ch07/cnn_pretrained_spacy_embeddings.hist.py
