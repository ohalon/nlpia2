from generate import generate_words
from model import RNNModel

corpus = data.Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='He'))
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='He'))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='He'))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='The'))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=.1))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=.2))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=.5))
import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel

corpus = Corpus('data/wikitext-2')
vocab = corpus.vocab
with open('model.pt', 'rb') as f:
    model = torch.load(f, map_location='cpu')
state_dict = model.state_dict()

model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
model.load_state_dict(state_dict)
' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=1))
more data/wikitext-2/train.txt
tail -n 100 data/wikitext-2/train.txt
!tail data/wikitext-2/train.txt -n 100
!tail data/wikitext-2/train.txt -n 120
!tail data/wikitext-2/train.txt -n 120 | head -n 20
!tail data/wikitext-2/train.txt -n 150 | head -n 20
!tail data/wikitext-2/train.txt -n 130 | head -n 20
!tail data/wikitext-2/train.txt -n 131 | head -n 20
!tail data/wikitext-2/train.txt -n 131 | head -n 3
!tail data/wikitext-2/train.txt -n 133 | head -n 3
!tail data/wikitext-2/train.txt -n 133 | head -n 10
dicts = list(jsonlines.open(fn))
fn = 'checkpoint_notes/grid_search/experiments_288.jsonl'
dicts = list(jsonlines.open(fn))
import jsonlines
dicts = list(jsonlines.open(fn))
df = pd.DataFrame(dicts)
imoprt pandas as pd
import pandas as pd
df = pd.DataFrame(dicts)
gru = gru[gru.columns[7:]]
gru = gru[gru.columns[7:]]
gru = df[df['rnn_type'] == 'GRU'].sort_values('test_loss')
grus = df[df['rnn_type'] == 'GRU'].sort_values('test_loss')
grus = grus[grus['epoch_num'] == 12]
grus
grus['num_parameters']
grus['parameters']
grus.columns
df = pd.read_csv('checkpoint_notes/random_search/experiment_grid_shuffled.csv')
df
ls checkpoint_notes
ls checkpoint_notes/grid_search/
df = pd.read_csv('checkpoint_notes/grid_search/experiment_plan_288_ch08.csv')
df
df = pd.read_csv('checkpoint_notes/grid_search/experiment_plan_288_ch08.csv', index=False)
df = pd.read_csv('checkpoint_notes/grid_search/experiment_plan_288_ch08.csv', index=None)
df = pd.read_csv('checkpoint_notes/grid_search/experiment_plan_288_ch08.csv', index_col=0)
df
ls checkpoint_notes/grid_search/
df = pd.read_csv('checkpoint_notes/grid_search/experiments_gru.csv', index_col=0)
df
df.sort_values('learned_parameters')
df['learned_parameters':]
df[['learned_parameters':]]
df[['learned_parameters']]
cols = 'learned_parameters rnn_type epochs lr num_layers dropout epoch_time test_loss'
cols = cols.split()
df[cols].round(2).sort_values('test_loss', ascending=False)
dict(zip(
    'learned_parameters  rnn_type  epochs   lr  num_layers  dropout  epoch_time  test_loss'.split(),
    'parameters  rnn_type  epochs   lr  layers  drop  time (s)  loss'.split()))
d = _
df.columns = [d.get(c, c) for c in df.columns]
cols = 'parameters  rnn_type  epochs   lr  layers  drop  time (s)  loss'.split()
cols = 'parameters  rnn_type  epochs   lr  layers  drop  sec/epoch  loss'.split()
dicts = list(jsonlines.open(fn))
df = pd.read_csv('checkpoint_notes/grid_search/experiments_gru.csv', index_col=0)
d = dict(zip(
    'learned_parameters  rnn_type  epochs   lr  num_layers  dropout  epoch_time  test_loss'.split(),
    'parameters  rnn_type  epochs   lr  layers  drop  time (s)  loss'.split()))
df.columns = [d.get(c, c) for c in df.columns]
df
cols = 'parameters  rnn_type  epochs   lr  layers  drop  sec/epoch  total_time loss'.split()
d = dict(zip(
    'learned_parameters  rnn_type  epochs   lr  num_layers  dropout  epoch_time  total_time test_loss'.split(),
    'parameters  rnn_type  epochs   lr  layers  drop  sec/epoch total_time  loss'.split()))
df.columns = [d.get(c, c) for c in df.columns]
dicts = list(jsonlines.open(fn))
df = pd.read_csv('checkpoint_notes/grid_search/experiments_gru.csv', index_col=0)
df.columns = [d.get(c, c) for c in df.columns]
df[cols]
df[cols].round(2).sort_values('test_loss')
df[cols].round(2).sort_values('loss')
df.to_csv('checkpoint_notes/grid_search/experiments_gru_rename_cols.csv')
df = pd.DataFrame(dicts)
df.columns = [d.get(c, c) for c in df.columns]
df[cols].round(2).sort_values('loss')
df
df.columns
df['total_time'] = df['sec/epoch'] * df['epochs']
df[cols].round(2).sort_values('loss')
df = pd.read_csv('checkpoint_notes/grid_search/experiment_plan_288_ch08.csv', index_col=0)
df.columns = [d.get(c, c) for c in df.columns]
df[cols].round(2).sort_values('loss')
df.columns
df = pd.DataFrame(dicts)
df
df.columns = [d.get(c, c) for c in df.columns]
df.columns
df[cols].round(2).sort_values('loss')
cols = 'parameters  rnn_type  epochs   lr  layers  drop  sec/epoch time loss'.split()
df[cols].round(2).sort_values('loss')
df.columns
hist -o -p -f checkpoint_notes/hyperparam_tables_ch08.hist.md
df = pd.read_csv('checkpoint_notes/grid_search/experiments_gru.csv', index_col=0)
df.columns = [d.get(c, c) for c in df.columns]
df[cols].round(2).sort_values('loss')
df['time'] = df['total_time']
df[cols].round(2).sort_values('loss')
df.iloc[195]
df[cols].round(2).sort_values('loss')
df.iloc[197]
fn = 'checkpoint_notes/grid_search/experiments_288.jsonl'
df = pd.DataFrame(list(jsonlines.open(fn)))
df.columns
!find . -name '*.jsonl'
df = pd.DataFrame(list(jsonlines.open('./checkpoint_notes/grid_search/experiments_200.jsonl')))
df.columns
ls checkpoint_notes/grid_search/
more checkpoint_notes/grid_search/experiments.txt
df = pd.DataFrame(list(jsonlines.open('./checkpoint_notes/grid_search/experiments.txt')))
df
df.columns
df = pd.DataFrame(list(jsonlines.open('./checkpoint_notes/random_search/experiments.txt')))
df = pd.DataFrame(list(jsonlines.open('./checkpoint_notes/random_search/experiment_grid_360.json')))
ls ./checkpoint_notes/random_search/
ls ./checkpoint_notes/
!git log --stat | grep jsonl
ls *.jsonl
ls
!git log --stat | grep jsonl -B6
hist -o -p -f checkpoint_notes/hyperparam_tables_ch08.hist.md
hist -f checkpoint_notes/hyperparam_tables_ch08.hist.py
