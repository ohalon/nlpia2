from torch import nn

class RNN(nn.Module):

    def __init__(self,
            vocab_size, hidden_size, output_size):  # <1>
        super().__init__()
        self.W_c2h = nn.Linear(
            vocab_size + hidden_size, hidden_size)  # <2>
        self.W_c2y = nn.Linear(vocab_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):  # <3>
        combined = torch.cat((x, hidden), axis=1)  # <4>
        hidden = self.W_c2h(combined)  # <5>
        y = self.W_c2y(combined)  # <6>
        y = self.softmax(y)
        return y, hidden  # <7>

import pandas as pd

from nlpia2.spacy_language_model import nlp

tagged_tokens = list(nlp('Hello world. Goodbye now!'))

interesting_tags = 'text dep_ head lang_ lemma_ pos_ sentiment'

interesting_tags = (interesting_tags +  'shape_ tag_').split()

pd.DataFrame([
        [getattr(t, a) for a in interesting_tags]
        for t in tagged_tokens],
    columns=interesting_tags)

from nlpia2.string_normalizers import Asciifier

asciify = Asciifier()

asciify("O’Néàl")

asciify("Çetin")

repo = 'tangibleai/nlpia2'  # <1>

filepath = 'src/nlpia2/data/surname-nationality.csv.gz'

url = f"https://gitlab.com/{repo}/-/raw/main/{filepath}"

df = pd.read_csv(url)  # <2>

df[['surname', 'nationality']].sort_values('surname').head(9)

df['nationality'].nunique()

sorted(df['nationality'].unique())

fraction_unique = {}

for i, g in df.groupby('nationality'):

    fraction_unique[i] = g['surname'].nunique() / len(g)

pd.Series(fraction_unique).sort_values().head(7)

arabic = [x.strip() for x in open('.nlpia2-data/names/Arabic.txt')]

arabic = pd.Series(sorted(arabic))

df.groupby('surname')

overlap = {}
for i, g in df.groupby('surname'):
    n = g['nationality'].nunique()
    if n > 1:
        overlap[i] = {'nunique': n, 'unique': list(g['nationality'].unique())}

overlap.sort_values('nunique', ascending=False)

class RNN(nn.Module):

def __init__(self, n_hidden=128, categories, char2i):  # <1>
    super().__init__()
    self.categories = categories
    self.n_categories = len(self.categories)  # <2>
    print(f'RNN.categories: {self.categories}')
    print(f'RNN.n_categories: {self.n_categories}')

def forward(self, x, hidden):  # <3>
    combined = torch.cat((x, hidden), 1)
    hidden = self.W_c2h(combined)
    y = self.W_c2y(combined)
    y = self.softmax(y)
    return y, hidden  # <4>

def train_sample(model, category_tensor, char_seq_tens,
                criterion=nn.NLLLoss(), lr=.005):

%run classify_name_nationality.py  # <1>

model.predict_category("Khalid")

predictions = topk_predictions(model, 'Khalid', topk=4)

predictions

predictions = topk_predictions(model, 'Khalid', topk=4)

predictions['likelihood'] = np.exp(predictions['log_loss'])

predictions

def predict_hidden(self, text="Khalid"):
   text_tensor = self.encode_one_hot_seq(text)
   with torch.no_grad():  # <1>
   hidden = self.hidden_init
       for i in range(text_tensor.shape[0]):  # <2>
           y, hidden = self(text_tensor[i], hidden)  # <3>
   return hidden

def predict_proba(self, text="Khalid"):
   text_tensor = self.encode_one_hot_seq(text)
   with torch.no_grad():
       hidden = self.hidden_init
       for i in range(text_tensor.shape[0]):
           y, hidden = self(text_tensor[i], hidden)
   return y  # <1>

def predict_category(self, text):
   tensor = self.encode_one_hot_seq(text)
   y = self.predict_proba(tensor)  # <1>
   pred_i = y.topk(1)[1][0].item()  # <2>
   return self.categories[pred_i]

text = 'Khalid'

pred_categories = []

pred_hiddens = []

for i in range(1, len(text) + 1):
   pred_hiddens.append(model.predict_hidden(text[:i]))  # <1>
   pred_categories.append(model.predict_category(text[:i]))

pd.Series(pred_categories, input_texts)

hiddens = [h[0].tolist() for h in hiddens]

df_hidden = pd.DataFrame(hidden_lists, index=list(text))

df_hidden = df_hidden.T.round(2)  # <1>

df_hidden

position = pd.Series(range(len(text)), index=df_hidden.index)

pd.DataFrame(position).T

df_hidden_raw.corrwith(position).sort_values()

lines = open('data/wikitext-2/train.txt').readlines()

for line in lines[:4]:
    print(line.rstrip()[:70])

from nlpia2.ch08.data import Corpus

corpus = Corpus('data/wikitext-2')

corpus.train

vocab = corpus.dictionary

[vocab.idx2word[i] for i in corpus.train[:7]]

def batchify_slow(x, batch_size=8, num_batches=5):
   batches = []
   for i in range(int(len(x)/batch_size)):
       if i > num_batches:
           break
       batches.append(x[i*batch_size:i*batch_size + batch_size])
   return batches

batches = batchify_slow(corpus.train)

batches

torch.stack(batches)

r = sigmoid(W_i2r.mm(x) + b_i2r +    W_h2r.mm(h) + b_h2r)  # <1>

z = sigmoid(W_i2z.mm(x) + b_i2z +    W_h2z.mm(h) + b_h2z)  # <2>

n =    tanh(W_i2n.mm(x) + b_i2n + r∗(W_h2n.mm(h) + b_h2n))  # <3>

def count_parameters(model, learned=True):
    return sum(
        p.numel() for p in model.parameters()  # <1>
        if not learned or p.requires_grad  # <2>
    )

import jsonlines  # <1>

with jsonlines.open('experiments.jsonl') as fin:
    lines = list(fin)

df = pd.DataFrame(lines)

df.to_csv('experiments.csv')

cols = 'learned_parameters rnn_type epochs lr num_layers'

cols += ' dropout epoch_time test_loss'

cols = cols.split()

df[cols].round(2).sort_values('test_loss', ascending=False)

df

from nlpia2.ch08.rnn_word.data import Corpus

corpus = Corpus('data/wikitext-2')

passage = corpus.train.numpy()[-89:-35]

' '.join([vocab.idx2word[i] for i in passage])

num_eos = sum([vocab.idx2word[i] == '<eos>' for i in

num_eos

num_unk = sum([vocab.idx2word[i] == '<unk>' for i in

num_unk

num_normal = sum([
    vocab.idx2word[i] not in ('<unk>', '<eos>')
    for i in corpus.train.numpy()])

num_normal

num_unk / (num_normal + num_eos + num_unk)

import torch

from preprocessing import Corpus

from generate import generate_words

from model import RNNModel

corpus = Corpus('data/wikitext-2')

vocab = corpus.dictionary

with open('model.pt', 'rb') as f:
   orig_model = torch.load(f, map_location='cpu')  # <1>

model = RNNModel('GRU', vocab=corpus.dictionary, num_layers=1)  # <2>

model.load_state_dict(orig_model.state_dict())

words = generate_words(
   model=model, vocab=vocab, prompt='The', temperature=.1)  # <3>

print(' '.join(w for w in words))
