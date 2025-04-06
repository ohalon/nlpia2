%run generate
hist
who
%run main.py
train_data
who
corpus
corpus.train
cd ~/code/team/exercises/
%run generate
who
generate_words(model=model, vocab=vocab, prompt='He')
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
# ' '.join(generate_words(model=model, vocab=vocab, prompt='The', temperature=1))
generate_words(model=model, vocab=vocab, prompt='He')
generate_text(model=model, vocab=vocab, prompt='He')
generate_text(model=model, vocab=vocab, prompt='The')
generate_text(model=model, vocab=vocab, prompt='President')
generate_text(model=model, vocab=vocab, prompt='The president of the United States')
generate_text(model=model, vocab=vocab, temperature=.2, prompt='President')
corpus
corpus.train
[vocab.idx2word[i] for i in corpus.train]
[vocab.idx2word[i] for i in corpus.train[:100]]
' '.join([vocab.idx2word[i] for i in corpus.train[:100]])
corpus.train
batchify
batchify?
for
for i in range(int(len(corpus.train)/batch_size)):
    print(i)
batch_size = 24
for i in range(int(len(corpus.train)/batch_size)):
    print(i)
for i in range(int(len(corpus.train)/batch_size)):
    print(i*batch_size)
for i in range(int(len(corpus.train)/batch_size)):
    print(i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]])
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    print(i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]])
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
pd.DataFrame(batches_desc)
import pandas as pd
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
pd.DataFrame(batches_desc)
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
pd.DataFrame(batches_desc, columns='batch_num start_tok_num start_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append((i, i*batch_size, vocab.idx2word[corpus.train[i*batch_size]]))
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size - 1]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos end_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size - 1]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_num end_tok_pos'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size - 1]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        (i+1)*batch_size - 1,
        vocab.idx2word[corpus.train[(i+1)*batch_size - 1]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size + batch_size,
        vocab.idx2word[corpus.train[i*batch_size + batch_size]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size + batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size + batch_size - ]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
batches_desc = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    batches_desc.append([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size + batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size + batch_size - 1]],
        ])
pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
batches = []
for i in range(int(len(corpus.train)/batch_size)):
    if i > 10:
        break
    print([
        i,
        i*batch_size,
        vocab.idx2word[corpus.train[i*batch_size]],
        i*batch_size + batch_size - 1,
        vocab.idx2word[corpus.train[i*batch_size + batch_size - 1]],
        ])
    batches.append(corpus.train[i*batch_size:i*batch_size + batch_size - 1])
    print(batches[-1])
# pd.DataFrame(batches_desc, columns='batch_num start_tok_pos start_tok end_tok_pos end_tok'.split())
def batchify1(x, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        print([
            i,
            i*batch_size,
            vocab.idx2word[x[i*batch_size]],
            i*batch_size + batch_size - 1,
            vocab.idx2word[x[i*batch_size + batch_size - 1]],
            ])
        batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
    return batches
x
x
x
x
def batchify1(x, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        print([
            i,
            i*batch_size,
            vocab.idx2word[x[i*batch_size]],
            i*batch_size + batch_size - 1,
            vocab.idx2word[x[i*batch_size + batch_size - 1]],
            ])
        batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
    return batches
batchify1(corpus.train)
def batchify1(x, batch_size=8, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        print([
            i,
            i*batch_size,
            vocab.idx2word[x[i*batch_size]],
            i*batch_size + batch_size - 1,
            vocab.idx2word[x[i*batch_size + batch_size - 1]],
            ])
        batches.append(x[i*batch_size:i*batch_size + batch_size - 1])
    return batches
batchify1(corpus.train, batch_size=24)
batchify1(corpus.train, batch_size=3)
batchify1(corpus.train)
def batchify1(x, batch_size=8, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        print([
            i,
            i*batch_size,
            vocab.idx2word[x[i*batch_size]],
            i*batch_size + batch_size - 1,
            vocab.idx2word[x[i*batch_size + batch_size - 1]],
            ])
        batches.append(x[i*batch_size:i*batch_size + batch_size])
    return batches
def batchify1(x, batch_size=8, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        print([
            i,
            i*batch_size,
            vocab.idx2word[x[i*batch_size]],
            i*batch_size + batch_size,
            vocab.idx2word[x[i*batch_size + batch_size]],
            ])
        batches.append(x[i*batch_size:i*batch_size + batch_size])
    return batches
batchify1(corpus.train)
def batchify1(x, batch_size=8, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i > num_batches:
            break
        batches.append(x[i*batch_size:i*batch_size + batch_size])
    return batches
batchify1(corpus.train)
batches = batchify1(corpus.train, num_batches=100)
torch.stack(batches)
torch.stack(batches).size
torch.stack(batches).size()
def batchify1(x, batch_size=8, num_batches=5):
    batches = []
    for i in range(int(len(x)/batch_size)):
        if i >= num_batches:
            break
        batches.append(x[i*batch_size:i*batch_size + batch_size])
    return torch.stack(batches)
batches = batchify1(corpus.train, num_batches=10, batch_size=3)
torch.stack(batches).size()
batches
batches.size()
batches = batchify1(corpus.train, num_batches=1000, batch_size=10)
batches.size()
train_epoch
train_epoch?
train_epoch(model, batches, ntokens=len(corpus.vocab.idx2word))
batches = batchify1(corpus.train, num_batches=1000, batch_size=20)
train_epoch(model, batches, ntokens=len(corpus.vocab.idx2word))
hist -o -p -f hist/ch08_batchify_rnn_input.hist.md
hist -f hist/ch08_batchify_rnn_input.hist.py
