ls - hal
ls models
ls - hal models
rm models / model_epochs_12_rnn_type_GRU_hidden_size_200_batch_size_20_bptt_35_num_layers_2
import torch
torch.load('models/model_epochs_12_rnn_type_GRU_hidden_size_200_batch_size_20_bptt_35_num_layers_1')
torch.load('models/model_epochs_12_rnn_type_GRU_hidden_size_200_batch_size_20_bptt_35_num_layers_1', map_location=torch.device('cpu'))
model = _
more generate.py
import argparse
import torch

import data
    args = parse_args()
    corpus = data.Corpus(args.data)

    token_id = corpus.vocab(args.prompt) if args.prompt else torch.randint
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    corpus = data.Corpus(args.data)

    token_id = corpus.vocab(args.prompt) if args.prompt else torch.randint
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus(args.data)

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.dicitonary))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus(args.data)

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.dicitonary))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
checkpoint = 'models/model_epochs_12_rnn_type_GRU_hidden_size_200_batch_size_20_bptt_35_num_layers_1'
checkpoint
    corpus = data.Corpus(args.data)

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.dicitonary))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus()

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.dicitonary))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.dicitonary))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.vocab))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.vocab), size=(,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = ''
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = 'Hello'
    token_id = corpus.vocab(prompt) if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(1111)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
corpus.vocab.word2idx
    corpus = data.Corpus('data/wikitext-2')

    prompt = 'Hello'
    token_id = corpus.vocab.idx2word(prompt) if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(1111)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = 'Hello'
    token_id = corpus.vocab.word2idx(prompt) if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(1111)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = 'Hello'
    token_id = corpus.vocab.word2idx[prompt] if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(1111)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    corpus = data.Corpus('data/wikitext-2')

    prompt = 'He'
    token_id = corpus.vocab.word2idx[prompt] if prompt else torch.randint(len(corpus.vocab), size=(1,))
    # Set the random seed manually for reproducibility.
    torch.manual_seed(1111)

    device = torch.device("cpu")

    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    model.eval?
more generate.py
hidden = model.init_hidden()
hidden = model.init_hidden(1)
input = torch.randint(len(corpus.vocab), (1, 1), dtype=torch.long).to(device)
word = prompt
word
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(args.temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
temperature = 0.01
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
word_weights
sum(word_weights)
input_tens
output
output.exp()
output.squeeze().exp()
output.squeeze().div(temperature).exp()
temperature = 1
output.squeeze().div(temperature).exp()
output.squeeze().div(.1).exp()
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(words) > max_words:
        break
word = prompt = 'He'
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(words) > max_words:
        break
word = prompt = 'He'
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<EOS>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words:
        break
output_words
word = prompt = 'He'
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words:
        break
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']
corpus
for doc in corpus:
    print(doc)
    break
for doc in corpus.train:
    print(doc)
    break
for i, doc in enumerate(corpus.train):
    print(i, doc)
    if i > 100:
        break
for i, doc in enumerate(corpus.train):
    print(i, corpus.vocab.idx2word[doc])
    if i > 100:
        break
hist
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']
word = prompt = 'He'
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    if word_idx == IDX_UNK:
        continue
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words or word_idx == IDX_EOS:
        break
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']
word = prompt = 'He'
temperature = 0.2
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    # if word_idx == IDX_UNK:
    #     continue
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words or word_idx == IDX_EOS:
        break
' '.join([w for w in output_words if w != '<unk>'])
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']
word = prompt = 'He'
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    # if word_idx == IDX_UNK:
    #     continue
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words or word_idx == IDX_EOS:
        break
' '.join([w for w in output_words if w != '<unk>'])
temperature = 1.0
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']
word = prompt = 'He'
output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    # if word_idx == IDX_UNK:
    #     continue
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words or word_idx == IDX_EOS:
        break
' '.join([w for w in output_words if w != '<unk>'])
IDX_UNK = corpus.vocab.word2idx['<unk>']
IDX_EOS = corpus.vocab.word2idx['<eos>']

output_words = []
output_words.append(word)
max_words = 1024
input_tens = torch.randint(corpus.vocab.word2idx[word], (1, 1), dtype=torch.long).to(device)
while word and word not in {'<eos>'}:
    output, hidden = model(input_tens, hidden)
    word_weights = output.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    # if word_idx == IDX_UNK:
    #     continue
    input_tens.fill_(word_idx)
    word = corpus.vocab.idx2word[word_idx]
    output_words.append(word)
    if len(output_words) > max_words or word_idx == IDX_EOS:
        break
word
hist - o - p - f rnn_word_generate.hist.md
hist - f rnn_word_generate.hist.py
