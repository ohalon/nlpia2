#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-08_Reduce,-Reuse,-Recycle-Recurrent-Neural-Networks`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-08_Reduce,-Reuse,-Recycle-Recurrent-Neural-Networks.adoc)

# #### .Recurrence in PyTorch

# In[ ]:


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


# #### .SpaCy tags tokens with RNNs

# In[ ]:


import pandas as pd
from nlpia2.spacy_language_model import nlp
tagged_tokens = list(nlp('Hello world. Goodbye now!'))
interesting_tags = 'text dep_ head lang_ lemma_ pos_ sentiment'
interesting_tags = (interesting_tags +  'shape_ tag_').split()
pd.DataFrame([
        [getattr(t, a) for a in interesting_tags]
        for t in tagged_tokens],
    columns=interesting_tags)


# #### .SpaCy tags tokens with RNNs

# In[ ]:


from nlpia2.string_normalizers import Asciifier
asciify = Asciifier()
asciify("O’Néàl")


# #### .SpaCy tags tokens with RNNs

# In[ ]:


asciify("Çetin")


# #### .Load the

# In[ ]:


repo = 'tangibleai/nlpia2'  # <1>
filepath = 'src/nlpia2/data/surname-nationality.csv.gz'
url = f"https://gitlab.com/{repo}/-/raw/main/{filepath}"
df = pd.read_csv(url)  # <2>
df[['surname', 'nationality']].sort_values('surname').head(9)


# #### .Load the

# In[ ]:


df['nationality'].nunique()


# #### .Load the

# In[ ]:


sorted(df['nationality'].unique())


# #### 

# In[ ]:


fraction_unique = {}
for i, g in df.groupby('nationality'):
    fraction_unique[i] = g['surname'].nunique() / len(g)
pd.Series(fraction_unique).sort_values().head(7)


# #### 

# In[ ]:


arabic = [x.strip() for x in open('.nlpia2-data/names/Arabic.txt')]
arabic = pd.Series(sorted(arabic))


# #### 

# In[ ]:


df.groupby('surname')
overlap = {}
for i, g in df.groupby('surname'):
    n = g['nationality'].nunique()
    if n > 1:
        overlap[i] = {'nunique': n, 'unique': list(g['nationality'].unique())}
overlap.sort_values('nunique', ascending=False)


# #### .Heart of an RNN

# In[ ]:


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


# #### .Heart of an RNN

# In[ ]:


def train_sample(model, category_tensor, char_seq_tens,
                criterion=nn.NLLLoss(), lr=.005):


# #### 

# In[ ]:


get_ipython().run_line_magic('run', 'classify_name_nationality.py  # <1>')


# #### 

# In[ ]:


model.predict_category("Khalid")


# #### 

# In[ ]:


predictions = topk_predictions(model, 'Khalid', topk=4)
predictions


# #### 

# In[ ]:


predictions = topk_predictions(model, 'Khalid', topk=4)
predictions['likelihood'] = np.exp(predictions['log_loss'])
predictions


# #### 

# In[ ]:


def predict_hidden(self, text="Khalid"):


# #### 

# In[ ]:


def predict_proba(self, text="Khalid"):
   text_tensor = self.encode_one_hot_seq(text)
   with torch.no_grad():
       hidden = self.hidden_init
       for i in range(text_tensor.shape[0]):
           y, hidden = self(text_tensor[i], hidden)
   return y  # <1>


# #### 

# In[ ]:


def predict_category(self, text):
   tensor = self.encode_one_hot_seq(text)
   y = self.predict_proba(tensor)  # <1>
   pred_i = y.topk(1)[1][0].item()  # <2>
   return self.categories[pred_i]


# #### 

# In[ ]:


text = 'Khalid'
pred_categories = []
pred_hiddens = []
for i in range(1, len(text) + 1):
   pred_hiddens.append(model.predict_hidden(text[:i]))  # <1>
   pred_categories.append(model.predict_category(text[:i]))
pd.Series(pred_categories, input_texts)


# #### 

# In[ ]:


hiddens = [h[0].tolist() for h in hiddens]
df_hidden = pd.DataFrame(hidden_lists, index=list(text))
df_hidden = df_hidden.T.round(2)  # <1>


# #### 

# In[ ]:


position = pd.Series(range(len(text)), index=df_hidden.index)
pd.DataFrame(position).T


# #### 

# In[ ]:


df_hidden_raw.corrwith(position).sort_values()


# #### 

# In[ ]:


lines = open('data/wikitext-2/train.txt').readlines()
for line in lines[:4]:
    print(line.rstrip()[:70])
from nlpia2.ch08.data import Corpus
corpus = Corpus('data/wikitext-2')
corpus.train


# #### 

# In[ ]:


vocab = corpus.dictionary
[vocab.idx2word[i] for i in corpus.train[:7]]


# #### 

# In[ ]:


def batchify_slow(x, batch_size=8, num_batches=5):
   batches = []
   for i in range(int(len(x)/batch_size)):
       if i > num_batches:
           break
       batches.append(x[i*batch_size:i*batch_size + batch_size])
   return batches
batches = batchify_slow(corpus.train)


# #### 

# In[ ]:


batches


# #### 

# In[ ]:


torch.stack(batches)


# #### 

# In[ ]:


r = sigmoid(W_i2r.mm(x) + b_i2r +    W_h2r.mm(h) + b_h2r)  # <1>
z = sigmoid(W_i2z.mm(x) + b_i2z +    W_h2z.mm(h) + b_h2z)  # <2>
n =    tanh(W_i2n.mm(x) + b_i2n + r∗(W_h2n.mm(h) + b_h2n))  # <3>


# #### 

# In[ ]:


def count_parameters(model, learned=True):
    return sum(
        p.numel() for p in model.parameters()  # <1>
        if not learned or p.requires_grad  # <2>
    )


# #### 

# In[ ]:


import jsonlines  # <1>
with jsonlines.open('experiments.jsonl') as fin:
    lines = list(fin)
df = pd.DataFrame(lines)
df.to_csv('experiments.csv')
cols = 'learned_parameters rnn_type epochs lr num_layers'
cols += ' dropout epoch_time test_loss'
cols = cols.split()
df[cols].round(2).sort_values('test_loss', ascending=False)


# #### 

# In[ ]:


df


# #### 

# In[ ]:


r = sigmoid(W_i2r.mm(x) + b_i2r +    W_h2r.mm(h) + b_h2r)
z = sigmoid(W_i2z.mm(x) + b_i2z +    W_h2z.mm(h) + b_h2z)
n =    tanh(W_i2n.mm(x) + b_i2n + r∗(W_h2n.mm(h) + b_h2n))

f = sigmoid(W_i2f.mm(x) + b_i2f + W_h2f.mm(h) + b_h2f)  # <1>
i = sigmoid(W_i2i.mm(x) + b_i2i + W_h2i.mm(h) + b_h2i)  # <2>
g = tanh(W_i2g.mm(x) + b_i2g + W_h2y.mm(h) + b_h2g)  # <3>
o = sigmoid(W_i2o.mm(x) + b_i2o + W_h2o.mm(h) + b_h2o)  # <4>
c = f*c + i*g  # <5>
h = o*tanh(c)
----
<1> LSTM forgetting gate (GRU reset gate)
<2> LSTM input relevance gate (GRU update gate)
<3> LSTM cell gate, notice the redundant biases b_i2i and b_h2i
<4> LSTM output gate
<5> cell state


=== Give your RNN a tuneup
// SUM: Hyperparameter tuning tricks, like increasing dropout percentages and reducing the number of learnable weights can help an RNN improve its accuracy and generalization in the real world. Bigger isn't always better. And to get your model out of a rut, increase the temperature during runtime.

As you learned in Chapter 7, hyperparameter tuning becomes more and more important as your neural networks get more and more complicated.
Your intuitions about layers, network capacity and training time will get fuzzier and fuzzier as the models get complicated.
RNNs are particularly intuitive.
To jumpstart your intuition we've trained dozens of different basic RNNs with different combinations of hyperparameters such as the number of layers and number of hidden units in each layer.
You can explore all the hyperparameters that you are curious about using the code in `nlpia2/ch08`.footnote:[The `hypertune.py` script in the `ch08/rnn_word` module within the `nlpia2` Python package https://gitlab.com/tangibleai/nlpia2/-/blob/main/src/nlpia2/ch08/rnn_word/hypertune.py]

[source,python]
----
import pandas as pd
import jsonlines

with jsonlines.open('experiments.jsonl') as fin:
    lines = list(fin)
df = pd.DataFrame(lines)
df.to_csv('experiments.csv')
cols = 'rnn_type epochs lr num_layers dropout epoch_time test_loss'
cols = cols.split()
df[cols].round(2).sort_values('test_loss').head(10)
----

[source,text]
----
    epochs   lr  num_layers  dropout  epoch_time  test_loss
37      12  2.0           2      0.2       35.43       5.23
28      12  2.0           1      0.0       22.66       5.23
49      32  0.5           2      0.0       32.35       5.22
57      32  0.5           2      0.2       35.50       5.22
38      12  2.0           3      0.2       46.14       5.21
50      32  0.5           3      0.0       37.36       5.20
52      32  2.0           1      0.0       22.90       5.10
55      32  2.0           5      0.0       56.23       5.09
53      32  2.0           2      0.0       32.49       5.06
54      32  2.0           3      0.0       38.78       5.04
----

It's a really exciting thing to explore the hyperspace of options like this and discover surprising tricks for building accurate models.
Surprisingly, for this RNN language model trained on a small subset of Wikipedia, you can get great results without maximizing the size and capacity of the model.
You can achieve better accuracy with a 3-layer RNN than with a 5-layer RNN.
You just need to start with an aggressive learning rate and keep the dropout to a minimum.
And the fewer layers you have the faster the model will train.

[TIP]
====


# #### 

# In[ ]:


from nlpia2.ch08.rnn_word.data import Corpus
corpus = Corpus('data/wikitext-2')
passage = corpus.train.numpy()[-89:-35]


# #### 

# In[ ]:


' '.join([vocab.idx2word[i] for i in passage])


# #### 

# In[ ]:


num_eos = sum([vocab.idx2word[i] == '<eos>' for i in


# #### 

# In[ ]:


num_eos


# #### 

# In[ ]:


num_unk = sum([vocab.idx2word[i] == '<unk>' for i in


# #### 

# In[ ]:


num_unk


# #### 

# In[ ]:


num_normal = sum([
    vocab.idx2word[i] not in ('<unk>', '<eos>')
    for i in corpus.train.numpy()])
num_normal


# #### 

# In[ ]:


num_unk / (num_normal + num_eos + num_unk)


# #### 

# In[ ]:


import torch
from preprocessing import Corpus
from generate import generate_words
from model import RNNModel


# #### 

# In[ ]:


print(' '.join(w for w in words))

