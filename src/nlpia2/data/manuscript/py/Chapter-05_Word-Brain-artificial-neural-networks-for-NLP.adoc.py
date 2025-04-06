def logistic(x, w=1., phase=0, gain=1):
   return gain / (1. + np.exp(-w * (x - phase)))

import pandas as pd

import numpy as np

import seaborn as sns

sns.set_style()

xy = pd.DataFrame(np.arange(-50, 50) / 10., columns=['x'])

for w, phase in zip([1, 3, 1, 1, .5], [0, 0, 2, -1, 0]):
   kwargs = dict(w=w, phase=phase)
   xy[f'{kwargs}'] = logistic(xy['x'], **kwargs)

xy.plot(grid="on", ylabel="y")

from collections import Counter

np.random.seed(451)

tokens = "green egg egg ham ham ham spam spam spam spam".split()

bow = Counter(tokens)

x = pd.Series(bow)

x

x1, x2, x3, x4 = x

x1, x2, x3, x4

w0 = np.round(.1 * np.random.randn(), 2)

w0

w1, w2, w3, w4 = (.1 * np.random.randn(len(x))).round(2)

w1, w2, w3, w4

x = np.array([1, x1, x2, x3, x4])  # <1>

w = np.array([w0, w1, w2, w3, w4])  # <2>

y = np.sum(w * x)  # <3>

y

threshold = 0.0

y = int(y > threshold)

y = logistic(x)

def neuron(x, w):
   z = sum(wi * xi for xi, wi in zip(x, w))  # <1>
   return z > 0  # <2>

def neuron(x, w):
   z = np.array(wi).dot(w)
   return z > 0

import pandas as pd

import numpy as np

pd.options.display.max_rows = 7

np.random.seed(451)

df = pd.read_csv(  # <1>
    'https://proai.org/baby-names-us.csv.gz')

df.to_csv(  # <2>
    'baby-names-us.csv.gz', compression='gzip')

df = df.sample(10_000)  # <3>

df.shape

df.groupby(['name', 'sex'])['count'].sum()[('Timothy',)]

df = df.set_index(['name', 'sex'])

groups = df.groupby(['name', 'sex'])

counts = groups['count'].sum()

counts

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    use_idf=False,  # <1>
    analyzer='char',
    ngram_range=(1, 3)  # <2>
    )

vectorizer

df = pd.DataFrame([list(tup) for tup in counts.index.values],
                  columns=['name', 'sex'])

df['count'] = counts.values

df

df['istrain'] = np.random.rand(len(df)) < .9

df

df.index = pd.MultiIndex.from_tuples(
    zip(df['name'], df['sex']), names=['name_', 'sex_'])

df

df_most_common = {}  # <1>

for name, group in df.groupby('name'):
    row_dict = group.iloc[group['count'].argmax()].to_dict()  # <2>
    df_most_common[(name, row_dict['sex'])] = row_dict

df_most_common = pd.DataFrame(df_most_common).T  # <3>

df_most_common['istest'] = ~df_most_common['istrain'].astype(bool)

df_most_common

df['istest'] = df_most_common['istest']

df['istest'] = df['istest'].fillna(False)

df['istrain'] = ~df['istest']

istrain = df['istrain']

df['istrain'].sum() / len(df)

df['istest'].sum() / len(df)

(df['istrain'].sum() + df['istest'].sum()) / len(df)

unique_names = df['name'][istrain].unique()

unique_names = df['name'][istrain].unique()

vectorizer.fit(unique_names)

vecs = vectorizer.transform(df['name'])

vecs

vecs = pd.DataFrame(vecs.toarray())

vecs.columns = vectorizer.get_feature_names_out()

vecs.index = df.index

vecs.iloc[:,:7]

vectorizer = TfidfVectorizer(analyzer='char',
   ngram_range=(1, 3), use_idf=False, lowercase=False)

vectorizer = vectorizer.fit(unique_names)

vecs = vectorizer.transform(df['name'])

vecs = pd.DataFrame(vecs.toarray())

vecs.columns = vectorizer.get_feature_names_out()

vecs.index = df.index

vecs.iloc[:,:5]

import pandas as pd

import re

dfs = pd.read_html('https://en.wikipedia.org/wiki/'
    + 'Comparison_of_deep-learning_software')

tabl = dfs[0]

bincols = list(tabl.loc[:, 'OpenMP support':].columns)

bincols += ['Open source', 'Platform', 'Interface']

dfd = {}

for i, row in tabl.iterrows():
   rowd = row.fillna('No').to_dict()
   for c in bincols:
       text = str(rowd[c]).strip().lower()
       tokens = re.split(r'\W+', text)
       tokens += '\*'
       rowd[c] = 0
       for kw, score in zip(
               'yes via roadmap no linux android python \*'.split(),
               [1, .9, .2, 0, 2, 2, 2, .1]):
           if kw in tokens:
               rowd[c] = score
               break
   dfd[i] = rowd

tabl = pd.DataFrame(dfd).T

scores = tabl[bincols].T.sum()  # <1>

tabl['Portability'] = scores

tabl = tabl.sort_values('Portability', ascending=False)

tabl = tabl.reset_index()

tabl[['Software', 'Portability']][:10]

import torch

class LogisticRegressionNN(torch.nn.Module):

model = LogisticRegressionNN(num_features=vecs.shape[1], num_outputs=1)

model

loss_func_train = torch.nn.BCELoss(
    weight=torch.Tensor(df[['count']][istrain].values))

loss_func_test = torch.nn.BCELoss(  # <1>
    weight=torch.Tensor(df[['count']][~istrain].values))

loss_func_train

from torch.optim import SGD

hyperparams = {'momentum': 0.001, 'lr': 0.02}  # <1>

optimizer = SGD(
    model.parameters(), **hyperparams)  # <2>

optimizer

X = vecs.values

y = (df[['sex']] == 'F').values

X_train = torch.Tensor(X[istrain])

X_test = torch.Tensor(X[~istrain])

y_train = torch.Tensor(y[istrain])

y_test = torch.Tensor(y[~istrain])

from tqdm import tqdm

num_epochs = 200

pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

for epoch in pbar_epochs:
     optimizer.zero_grad()  # <1>
     outputs = model(X_train)
     loss_train = loss_func_train(outputs, y_train)  # <2>
     loss_train.backward()  # <3>
     optimizer.step()  # <4>

def make_array(x):
    if hasattr(x, 'detach'):
        return torch.squeeze(x).detach().numpy()
    return x

def measure_binary_accuracy(y_pred, y):
    y_pred = make_array(y_pred).round()
    y = make_array(y).round()
    num_correct = (y_pred == y).sum()
    return num_correct / len(y)

X = vectorizer.transform(
    ['John', 'Greg', 'Vishvesh',  # <1>

model(torch.Tensor(X.todense()))
