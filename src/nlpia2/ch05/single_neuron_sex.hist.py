df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
import pandas as pd
df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
df.to_csv('baby-names-us.csv.gz')
ls -hal
ls -hal /home/hobs/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz
rm baby-names-us.csv.gz
len(df)
10_000 / len(df)
>>> df = df.sample(10_000)  # <3>
df
>>> df = df.set_index(['name', 'sex'])
>>> groups = df.groupby(['name', 'sex'])
>>> counts = groups['count'].sum()
>>> counts
>>> counts[('Timothy',)]
>>> counts[('Tim',)]
df.groupby(['name', 'sex'])['count'].sum()[('Tim',)]
>>> import pandas as pd
>>> import numpy as np
>>> pd.options.display.max_rows = 7
>>> np.random.seed(451)
>>> df = pd.read_csv(  # <1>
...     'https://proai.org/baby-names-us.csv.gz')
df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> np.random.seed(451)
df.sample(10_000).groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
df.groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
!find ~/code/tangibleai/nlpia2/ -name '*name*csv*'
ls -hal /home/hobs/code/tangibleai/nlpia2/.nlpia2-data/baby-names-region.csv.gz
len(df)
>>> np.random.seed(451)
df = df.sample(10_000)
df.to_csv('~/code/tangibleai/nlpia2/src/nlpia2/data/baby-names-us-10k.csv.gz', compression='gzip')
>>> df.groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
>>> df = df.set_index(['name', 'sex'])
>>> groups = df.groupby(['name', 'sex'])
>>> counts = groups['count'].sum()
>>> counts
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(use_idf=False,  # <1>
...     analyzer='char', ngram_range=(1, 3)) 
>>> vectorizer
>>> df = pd.DataFrame([list(tup) for tup in counts.index.values],
...                   columns=['name', 'sex'])
>>> df['count'] = counts.values
>>> df
df['istrain'] = np.random.rand(len(df)) < .9
df
>>> df.index = pd.MultiIndex.from_tuples(
...     zip(df['name'], df['sex']), names=['name_', 'sex_'])
>>> df
>>> df_most_common = {}  # <1>
>>> for name, group in df.groupby('name'):
...     row_dict = group.iloc[group['count'].argmax()].to_dict() # <2>
...     df_most_common[(name, row_dict['sex'])] = row_dict
>>> df_most_common = pd.DataFrame(df_most_common).T  # <3>
>>> df_most_common['istest'] = ~df_most_common['istrain'].astype(bool)
>>> df_most_common
>>> df['istest'] = df_most_common['istest']
>>> df['istest'] = df['istest'].fillna(False)
>>> df['istrain'] = ~df['istest']
>>> istrain = df['istrain']
>>> df['istrain'].sum() / len(df)
df['istest'].sum() / len(df)
(df['istrain'] + df['istest']).sum() / len(df)
(df['istrain'].sum() + df['istest'].sum()) / len(df)
>>> unique_names = df['name'][istrain].unique()
>>> unique_names = df['name'][istrain].unique()
>>> vectorizer.fit(unique_names)
>>> vecs = vectorizer.transform(df['name'])
>>> vecs
>>> vecs = pd.DataFrame(vecs.toarray())
>>> vecs.columns = vectorizer.get_feature_names_out()
>>> vecs.index = df.index
>>> vecs.iloc[:,:7]
>>> vectorizer = TfidfVectorizer(analyzer='char',
...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
>>> vecs = vectorizer.fit_transform(unique_names)
>>> vecs = pd.DataFrame(vecs.toarray())
>>> vecs.columns = vectorizer.get_feature_names_out()
>>> vecs.index = df.index
>>> vecs.iloc[:,:5]
>>> vectorizer = TfidfVectorizer(analyzer='char',
...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
>>> vectorizer = vectorizer.fit(unique_names)
>>> vecs = vectorizer.transform(df['name'])
>>> vecs = pd.DataFrame(vecs.toarray())
>>> vecs.columns = vectorizer.get_feature_names_out()
>>> vecs.index = df.index
>>> vecs.iloc[:,:5]
>>> import pandas as pd
>>> import re

>>> dfs = pd.read_html('https://en.wikipedia.org/wiki/'
...     + 'Comparison_of_deep-learning_software')
>>> df = dfs[0]
df
>>> bincols = list(df.loc[:, 'OpenMP support':].columns)
>>> bincols += ['Open source', 'Platform', 'Interface']
>>> dfd = {}
>>> for i, row in df.iterrows():
...    rowd = row.fillna('No').to_dict()
...    for c in bincols:
...        text = str(rowd[c]).strip().lower()
...        tokens = re.split(r'\W+', text)
...        tokens += '\*'
...        rowd[c] = 0
...        for kw, score in zip(
...                'yes via roadmap no linux android python \*'.split(),
...                [1, .9, .2, 0, 2, 2, 2, .1]):
...            if kw in tokens:
...                rowd[c] = score
...                break
...    dfd[i] = rowd
>>> df = pd.DataFrame(dfd).T
>>> scores = df[bincols].T.sum()
>>> df['Portability'] = scores
>>> df = df.sort_values('Portability', ascending=False)

>>> # actively developed, open source, supports Linux, python API:
>>> df = df.reset_index()
>>> df[['Software', 'Portability']][:10]
vecs
model
>>> import torch 
>>> class LogisticRegressionNN(torch.nn.Module):

...    def __init__(self, num_features, num_outputs=1):
...         super().__init__()
...         self.linear = torch.nn.Linear(num_features, num_outputs)

...    def forward(self, X):
...        return torch.sigmoid(self.linear(X))
>>> model = LogisticRegressionNN(num_features=vecs.shape[1], num_outputs=1)
model
>>> X = vecs.values
>>> y = (df[['sex']] == 'F').values
>>> X_train = torch.Tensor(X[istrain])
>>> X_test = torch.Tensor(X[~istrain])
>>> y_train = torch.Tensor(y[istrain])
>>> y_test = torch.Tensor(y[~istrain])
df.columns
df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> np.random.seed(451)
df = df.sample(10_000)
>>> df = df.set_index(['name', 'sex'])
>>> groups = df.groupby(['name', 'sex'])
>>> counts = groups['count'].sum()
>>> counts
>>> df = pd.DataFrame([list(tup) for tup in counts.index.values],
...                   columns=['name', 'sex'])
>>> df['count'] = counts.values
>>> df
>>> df['istrain'] = np.random.rand(len(df)) < .9
>>> df
>>> df.index = pd.MultiIndex.from_tuples(
...     zip(df['name'], df['sex']), names=['name_', 'sex_'])
>>> df
>>> df_most_common = {}  # <1>
>>> for name, group in df.groupby('name'):
...     row_dict = group.iloc[group['count'].argmax()].to_dict() # <2>
...     df_most_common[(name, row_dict['sex'])] = row_dict
>>> df_most_common = pd.DataFrame(df_most_common).T  # <3>
>>> df_most_common['istest'] = ~df_most_common['istrain'].astype(bool)
>>> df_most_common
>>> df['istest'] = df_most_common['istest']
>>> df['istest'] = df['istest'].fillna(False)
>>> df['istrain'] = ~df['istest']
>>> istrain = df['istrain']
>>> df['istrain'].sum() / len(df)
>>> unique_names = df['name'][istrain].unique()
>>> unique_names = df['name'][istrain].unique()
>>> vectorizer.fit(unique_names)
>>> vecs = vectorizer.transform(df['name'])
>>> vecs
>>> vecs = pd.DataFrame(vecs.toarray())
>>> vecs.columns = vectorizer.get_feature_names_out()
>>> vecs.index = df.index
>>> vecs.iloc[:,:7]
>>> vectorizer = TfidfVectorizer(analyzer='char',
...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
>>> vectorizer = vectorizer.fit(unique_names)
>>> vecs = vectorizer.transform(df['name'])
>>> vecs = pd.DataFrame(vecs.toarray())
>>> vecs.columns = vectorizer.get_feature_names_out()
>>> vecs.index = df.index
>>> vecs.iloc[:,:5]
>>> X = vecs.values
>>> y = (df[['sex']] == 'F').values
>>> X_train = torch.Tensor(X[istrain])
>>> X_test = torch.Tensor(X[~istrain])
>>> y_train = torch.Tensor(y[istrain])
>>> y_test = torch.Tensor(y[~istrain])
>>> import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
range(num_epochs)
>>> from tqdm import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam?
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
hist -o -p
loss_train
loss_train = torch.nn.BCELoss(weight=torch.Tensor(df[['count']][istrain].values))
loss_train
>>> from torch.optim import SGD
>>> hyperparams = {'momentum': 0.001, 'lr': 0.02}  # <1>
>>> optimizer = SGD(model.parameters(), **hyperparams)  # <2>
>>> optimizer
model.parameters()
>>> from torch.optim import SGD
>>> hyperparams = {'momentum': 0.001, 'lr': 0.02}  # <1>
>>> optimizer = SGD(list(model.parameters()), **hyperparams)  # <2>
>>> optimizer
>>> from tqdm import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
>>> loss_func_train = torch.nn.BCELoss(
...     weight=torch.Tensor(df[['count']][istrain].values))
>>> loss_func_train
>>> from tqdm import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
>>> from tqdm import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
>>> from tqdm import tqdm
>>> num_epochs = 200
>>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)

>>> for epoch in pbar_epochs:
...      optimizer.zero_grad() # <1>
...      outputs = model(X_train)
...      loss_train = loss_func_train(outputs, y_train) # <2>
...      loss_train.backward() # <3>
...      optimizer.step() # <4>
>>> def make_array(x):
...     if hasattr(x, 'detach'):
...         return torch.squeeze(x).detach().numpy()
...     return x
def measure_binary_accuracy(y_pred, y):
    y_pred = make_array(y_pred).round()
    y = make_array(y).round()
    num_correct = (y_pred == y).sum()
    return num_correct / len(y)
>>> def measure_binary_accuracy(y_pred, y):
...     y_pred = make_array(y_pred).round()
...     y = make_array(y).round()
...     num_correct = (y_pred == y).sum()
...     return num_correct / len(y)
>>> loss_func_test = torch.nn.BCELoss(  # <1>
...     weight=torch.Tensor(df[['count']][~istrain].values))
for epoch in range(num_epochs):
    optimizer.zero_grad() 
    outputs = model(X_train)
    loss_train = loss_func_train(outputs, y_train) 
    loss_train.backward() 
    epoch_loss_train = loss_train.item()
    optimizer.step() 
    outputs_test = model(X_test)
    loss_test = loss_func_test(outputs_test, y_test).item()
    accuracy_test = measure_binary_accuracy(outputs_test, y_test)
    if epoch % 20 == 19:
        print((f'Epoch {epoch}: 
           loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f},
           accuracy_test: {accuracy_test:.4f}'))
for epoch in range(num_epochs):
    optimizer.zero_grad() 
    outputs = model(X_train)
    loss_train = loss_func_train(outputs, y_train) 
    loss_train.backward() 
    epoch_loss_train = loss_train.item()
    optimizer.step() 
    outputs_test = model(X_test)
    loss_test = loss_func_test(outputs_test, y_test).item()
    accuracy_test = measure_binary_accuracy(outputs_test, y_test)
    if epoch % 20 == 19:
        print(f'Epoch {epoch}:'
            f'loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f},
            f'accuracy_test: {accuracy_test:.4f}')
for epoch in range(num_epochs):
    optimizer.zero_grad() 
    outputs = model(X_train)
    loss_train = loss_func_train(outputs, y_train) 
    loss_train.backward() 
    epoch_loss_train = loss_train.item()
    optimizer.step() 
    outputs_test = model(X_test)
    loss_test = loss_func_test(outputs_test, y_test).item()
    accuracy_test = measure_binary_accuracy(outputs_test, y_test)
    if epoch % 20 == 19:
        print(f'Epoch {epoch}:'
            f' loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f}'
            f' accuracy_test: {accuracy_test:.4f}')
for epoch in range(num_epochs):
    optimizer.zero_grad() 
    outputs = model(X_train)
    loss_train = loss_func_train(outputs, y_train) 
    loss_train.backward() 
    epoch_loss_train = loss_train.item()
    optimizer.step() 
    outputs_test = model(X_test)
    loss_test = loss_func_test(outputs_test, y_test).item()
    accuracy_test = measure_binary_accuracy(outputs_test, y_test)
    if epoch % 20 == 19:
        print(f'Epoch {epoch}:'
            f' loss_train/test: {loss_train.item():.4f}/{loss_test:.4f},'
            f' accuracy_test: {accuracy_test:.4f}')
vectorizer.transform(['Cason'])
x_cason = vectorizer.transform(['Cason'])
model(x_cason)
X_train
model(torch.Tensor(x_cason))
model(torch.Tensor(x_cason.todense()))
x_cason = vectorizer.transform(['John'])
model(torch.Tensor(x_cason.todense()))
x_cason = vectorizer.transform(['Maria'])
model(torch.Tensor(x_cason.todense()))
x_cason = vectorizer.transform(['John', 'Vish', 'Sarah', '])
x_cason = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
x = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
X = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
model(torch.Tensor(X.todense()))
hist -o -p
x_vish = vectorizer.transform(['Vish'])
x_vish
[v for v in x_vish.fillna(0) if v]
[v for v in x_vish.todense().fillna(0) if v]
[v for v in x_vish if v]
[v for v in x_vish[0] if v]
[v for v in x_vish.todense()[0] if v]
x_vish.todense()
x_vish.todense()[0]
x_vish.todense()[0,:]
x_vish.todense().flatten()
x_vish.todense().ravel()
x_vish = vectorizer.transform(['Vishvesh'])
model(torch.Tensor(x_vish.todense()))
>>> X = vectorizer.transform(
...     ['John', 'Vishvesh', 'Greg',  # <1>
...     'Sarah', 'Ruby'])  # <2>
>>> model(torch.Tensor(X.todense()))
>>> X = vectorizer.transform(
...     ['John', 'Greg', 'Vishvesh',  # <1>
...      'Sarah', 'Ruby', 'Carlana'])  # <2>
>>> model(torch.Tensor(X.todense()))
>>> X = vectorizer.transform(
...     ['John', 'Greg', 'Vishvesh',  # <1>
...      'Ruby', 'Sarah', 'Carlana'])  # <2>
>>> model(torch.Tensor(X.todense()))
>>> X = vectorizer.transform(
...     ['John', 'Greg', 'Vishvesh',  # <1>
...      'Ruby', 'Carlana', 'Sarah'])  # <2>
>>> model(torch.Tensor(X.todense()))
hist -o -p -f ~/code/tangibleai/nlpia2/src/nlpia2/ch05/single_neuron_sex.hist.ipy.md
hist -f ~/code/tangibleai/nlpia2/src/nlpia2/ch05/single_neuron_sex.hist.py
