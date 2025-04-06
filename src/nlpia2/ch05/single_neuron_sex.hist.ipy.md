>>> df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> import pandas as pd
>>> df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> df.to_csv('baby-names-us.csv.gz')
>>> ls -hal
>>> ls -hal /home/hobs/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz
>>> rm baby-names-us.csv.gz
>>> len(df)
6241373
>>> 10_000 / len(df)
0.0016022115646669411
>>> >>> df = df.sample(10_000)  # <3>
>>> df
        region sex  year       name  count      freq
5582171     TX   M  2018    Rayland      5  0.000002
2302941     LA   F  1955     Joelle      5  0.000001
2945314     MN   F  1993       Rosa      7  0.000002
2617063     MD   M  1919      Louis     90  0.000046
3976840     NY   F  1946   Adrienne    124  0.000041
...        ...  ..   ...        ...    ...       ...
1456450     GA   M  2010     Gideon     14  0.000004
3726105     NJ   F  1999    Corinne     22  0.000007
4805894     PA   M  1950  Alexander    107  0.000032
6181031     WV   M  2003     Connor     42  0.000013
537361      CA   F  1999     Zulema     20  0.000006

[10000 rows x 6 columns]
>>> >>> df = df.set_index(['name', 'sex'])
... >>> groups = df.groupby(['name', 'sex'])
... >>> counts = groups['count'].sum()
... >>> counts
...
name     sex
Aadhya   F        5
Aaliyah  F       14
Aaron    M      783
Aarron   M        5
Aarya    F        8
               ... 
Zoya     F        9
Zulema   F       20
Zuri     F       64
Zyair    M        8
Zyaire   M       10
Name: count, Length: 4288, dtype: int64
>>> >>> counts[('Timothy',)]
sex
M    1572
Name: count, dtype: int64
>>> >>> counts[('Tim',)]
sex
M    668
Name: count, dtype: int64
>>> df.groupby(['name', 'sex'])['count'].sum()[('Tim',)]
sex
M    668
Name: count, dtype: int64
>>> >>> import pandas as pd
... >>> import numpy as np
... >>> pd.options.display.max_rows = 7
...
>>> >>> np.random.seed(451)
... >>> df = pd.read_csv(  # <1>
... ...     'https://proai.org/baby-names-us.csv.gz')
...
>>> df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> >>> np.random.seed(451)
>>> df.sample(10_000).groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
sex
F       5
M    3538
Name: count, dtype: int64
>>> df.groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
sex
F       1761
M    1071634
Name: count, dtype: int64
>>> !find ~/code/tangibleai/nlpia2/ -name '*name*csv*'
>>> ls -hal /home/hobs/code/tangibleai/nlpia2/.nlpia2-data/baby-names-region.csv.gz
>>> len(df)
6241373
>>> >>> np.random.seed(451)
>>> df = df.sample(10_000)
>>> df.to_csv('~/code/tangibleai/nlpia2/src/nlpia2/data/baby-names-us-10k.csv.gz', compression='gzip')
>>> >>> df.groupby(['name', 'sex'])['count'].sum()[('Timothy',)]
sex
F       5
M    3538
Name: count, dtype: int64
>>> >>> df = df.set_index(['name', 'sex'])
... >>> groups = df.groupby(['name', 'sex'])
... >>> counts = groups['count'].sum()
... >>> counts
...
name    sex
Aaden   M      51
Aahana  F      26
Aahil   M       5
               ..
Zvi     M       5
Zya     F       8
Zylah   F       5
Name: count, Length: 4238, dtype: int64
>>> >>> from sklearn.feature_extraction.text import TfidfVectorizer
... >>> vectorizer = TfidfVectorizer(use_idf=False,  # <1>
... ...     analyzer='char', ngram_range=(1, 3)) 
... >>> vectorizer
...
TfidfVectorizer(analyzer='char', ngram_range=(1, 3), use_idf=False)
>>> >>> df = pd.DataFrame([list(tup) for tup in counts.index.values],
... ...                   columns=['name', 'sex'])
... >>> df['count'] = counts.values
... >>> df
...
        name sex  count
0      Aaden   M     51
1     Aahana   F     26
2      Aahil   M      5
...      ...  ..    ...
4235     Zvi   M      5
4236     Zya   F      8
4237   Zylah   F      5

[4238 rows x 3 columns]
>>> df['istrain'] = np.random.rand(len(df)) < .9
>>> df
        name sex  count  istrain
0      Aaden   M     51     True
1     Aahana   F     26     True
2      Aahil   M      5     True
...      ...  ..    ...      ...
4235     Zvi   M      5     True
4236     Zya   F      8     True
4237   Zylah   F      5     True

[4238 rows x 4 columns]
>>> >>> df.index = pd.MultiIndex.from_tuples(
... ...     zip(df['name'], df['sex']), names=['name_', 'sex_'])
... >>> df
...
               name sex  count  istrain
name_  sex_                            
Aaden  M      Aaden   M     51     True
Aahana F     Aahana   F     26     True
Aahil  M      Aahil   M      5     True
...             ...  ..    ...      ...
Zvi    M        Zvi   M      5     True
Zya    F        Zya   F      8     True
Zylah  F      Zylah   F      5     True

[4238 rows x 4 columns]
>>> >>> df_most_common = {}  # <1>
... >>> for name, group in df.groupby('name'):
... ...     row_dict = group.iloc[group['count'].argmax()].to_dict() # <2>
... ...     df_most_common[(name, row_dict['sex'])] = row_dict
... >>> df_most_common = pd.DataFrame(df_most_common).T  # <3>
...
>>> >>> df_most_common['istest'] = ~df_most_common['istrain'].astype(bool)
... >>> df_most_common
...
            name sex count istrain  istest
Aaden  M   Aaden   M    51    True   False
Aahana F  Aahana   F    26    True   False
Aahil  M   Aahil   M     5    True   False
...          ...  ..   ...     ...     ...
Zvi    M     Zvi   M     5    True   False
Zya    F     Zya   F     8    True   False
Zylah  F   Zylah   F     5    True   False

[4025 rows x 5 columns]
>>> >>> df['istest'] = df_most_common['istest']
... >>> df['istest'] = df['istest'].fillna(False)
... >>> df['istrain'] = ~df['istest']
... >>> istrain = df['istrain']
... >>> df['istrain'].sum() / len(df)
...
0.9091552619159982
>>> df['istest'].sum() / len(df)
0.09084473808400188
>>> (df['istrain'] + df['istest']).sum() / len(df)
1.0
>>> (df['istrain'].sum() + df['istest'].sum()) / len(df)
1.0
>>> >>> unique_names = df['name'][istrain].unique()
... >>> unique_names = df['name'][istrain].unique()
... >>> vectorizer.fit(unique_names)
... >>> vecs = vectorizer.transform(df['name'])
... >>> vecs
...
<4238x2855 sparse matrix of type '<class 'numpy.float64'>'
	with 59959 stored elements in Compressed Sparse Row format>
>>> >>> vecs = pd.DataFrame(vecs.toarray())
... >>> vecs.columns = vectorizer.get_feature_names_out()
... >>> vecs.index = df.index
... >>> vecs.iloc[:,:7]
...
                    a        aa  aac       aad       aah  aak  aal
name_  sex_                                                       
Aaden  M     0.534522  0.267261  0.0  0.267261  0.000000  0.0  0.0
Aahana F     0.769800  0.192450  0.0  0.000000  0.192450  0.0  0.0
Aahil  M     0.534522  0.267261  0.0  0.000000  0.267261  0.0  0.0
...               ...       ...  ...       ...       ...  ...  ...
Zvi    M     0.000000  0.000000  0.0  0.000000  0.000000  0.0  0.0
Zya    F     0.408248  0.000000  0.0  0.000000  0.000000  0.0  0.0
Zylah  F     0.288675  0.000000  0.0  0.000000  0.000000  0.0  0.0

[4238 rows x 7 columns]
>>> >>> vectorizer = TfidfVectorizer(analyzer='char',
... ...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
... >>> vecs = vectorizer.fit_transform(unique_names)
... >>> vecs = pd.DataFrame(vecs.toarray())
... >>> vecs.columns = vectorizer.get_feature_names_out()
... >>> vecs.index = df.index
... >>> vecs.iloc[:,:5]
...
>>> >>> vectorizer = TfidfVectorizer(analyzer='char',
... ...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
... >>> vectorizer = vectorizer.fit(unique_names)
... >>> vecs = vectorizer.transform(df['name'])
...
>>> >>> vecs = pd.DataFrame(vecs.toarray())
... >>> vecs.columns = vectorizer.get_feature_names_out()
... >>> vecs.index = df.index
... >>> vecs.iloc[:,:5]
...
                    A        Aa       Aad       Aah  Aal
name_  sex_                                             
Aaden  M     0.288675  0.288675  0.288675  0.000000  0.0
Aahana F     0.218218  0.218218  0.000000  0.218218  0.0
Aahil  M     0.288675  0.288675  0.000000  0.288675  0.0
...               ...       ...       ...       ...  ...
Zvi    M     0.000000  0.000000  0.000000  0.000000  0.0
Zya    F     0.000000  0.000000  0.000000  0.000000  0.0
Zylah  F     0.000000  0.000000  0.000000  0.000000  0.0

[4238 rows x 5 columns]
>>> >>> import pandas as pd
... >>> import re
... 
... >>> dfs = pd.read_html('https://en.wikipedia.org/wiki/'
... ...     + 'Comparison_of_deep-learning_software')
... >>> df = dfs[0]
...
>>> df
                                Software                                            Creator  ...  Parallel execution (multi node)  Actively developed
0                                  BigDL                                  Jason Dai (Intel)  ...                              NaN                 NaN
1                                  Caffe                Berkeley Vision and Learning Center  ...                                ?               No[6]
2                                Chainer                                 Preferred Networks  ...                              Yes               No[7]
..                                   ...                                                ...  ...                              ...                 ...
20                                 Torch  Ronan Collobert, Koray Kavukcuoglu, Clement Fa...  ...                          Yes[64]                  No
21  Wolfram Mathematica 10[74] and later                                   Wolfram Research  ...                          Yes[76]                 Yes
22                              Software                                            Creator  ...  Parallel execution (multi node)  Actively developed

[23 rows x 19 columns]
>>> >>> bincols = list(df.loc[:, 'OpenMP support':].columns)
... >>> bincols += ['Open source', 'Platform', 'Interface']
... >>> dfd = {}
... >>> for i, row in df.iterrows():
... ...    rowd = row.fillna('No').to_dict()
... ...    for c in bincols:
... ...        text = str(rowd[c]).strip().lower()
... ...        tokens = re.split(r'\W+', text)
... ...        tokens += '\*'
... ...        rowd[c] = 0
... ...        for kw, score in zip(
... ...                'yes via roadmap no linux android python \*'.split(),
... ...                [1, .9, .2, 0, 2, 2, 2, .1]):
... ...            if kw in tokens:
... ...                rowd[c] = score
... ...                break
... ...    dfd[i] = rowd
...
>>> >>> df = pd.DataFrame(dfd).T
... >>> scores = df[bincols].T.sum()
... >>> df['Portability'] = scores
... >>> df = df.sort_values('Portability', ascending=False)
... 
... >>> # actively developed, open source, supports Linux, python API:
... >>> df = df.reset_index()
... >>> df[['Software', 'Portability']][:10]
...
                                Software Portability
0                                PyTorch        15.9
1                             TensorFlow        14.2
2                           Apache MXNet        14.2
..                                   ...         ...
7                           Apache SINGA          11
8   Wolfram Mathematica 10[74] and later          11
9                                Chainer          11

[10 rows x 2 columns]
>>> vecs
                    A        Aa       Aad       Aah  Aal  Aan  Aar   Ab  Abb  Abd  Abe  ...   zm  zmi  zmy   zo   zr  zra   zv  zvi   zz  zze  zzi
name_  sex_                                                                             ...                                                       
Aaden  M     0.288675  0.288675  0.288675  0.000000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
Aahana F     0.218218  0.218218  0.000000  0.218218  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
Aahil  M     0.288675  0.288675  0.000000  0.288675  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
...               ...       ...       ...       ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
Zvi    M     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
Zya    F     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
Zylah  F     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

[4238 rows x 3663 columns]
>>> model
>>> >>> import torch 
... >>> class LogisticRegressionNN(torch.nn.Module):
... 
... ...    def __init__(self, num_features, num_outputs=1):
... ...         super().__init__()
... ...         self.linear = torch.nn.Linear(num_features, num_outputs)
... 
... ...    def forward(self, X):
... ...        return torch.sigmoid(self.linear(X))
... >>> model = LogisticRegressionNN(num_features=vecs.shape[1], num_outputs=1)
...
>>> model
LogisticRegressionNN(
  (linear): Linear(in_features=3663, out_features=1, bias=True)
)
>>> >>> X = vecs.values
... >>> y = (df[['sex']] == 'F').values
... >>> X_train = torch.Tensor(X[istrain])
... >>> X_test = torch.Tensor(X[~istrain])
... >>> y_train = torch.Tensor(y[istrain])
... >>> y_test = torch.Tensor(y[~istrain])
...
>>> df.columns
Index(['index', 'Software', 'Creator', 'Initial release',
       'Software license[a]', 'Open source', 'Platform', 'Written in',
       'Interface', 'OpenMP support', 'OpenCL support', 'CUDA support',
       'ROCm support[1]', 'Automatic differentiation[2]',
       'Has pretrained models', 'Recurrent nets', 'Convolutional nets',
       'RBM/DBNs', 'Parallel execution (multi node)', 'Actively developed',
       'Portability'],
      dtype='object')
>>> df = pd.read_csv('~/Dropbox/Public/data/.nlpia2-data/baby-names-us.csv.gz')
>>> >>> np.random.seed(451)
>>> df = df.sample(10_000)
>>> >>> df = df.set_index(['name', 'sex'])
... >>> groups = df.groupby(['name', 'sex'])
... >>> counts = groups['count'].sum()
... >>> counts
...
name    sex
Aaden   M      51
Aahana  F      26
Aahil   M       5
               ..
Zvi     M       5
Zya     F       8
Zylah   F       5
Name: count, Length: 4238, dtype: int64
>>> >>> df = pd.DataFrame([list(tup) for tup in counts.index.values],
... ...                   columns=['name', 'sex'])
... >>> df['count'] = counts.values
... >>> df
...
        name sex  count
0      Aaden   M     51
1     Aahana   F     26
2      Aahil   M      5
...      ...  ..    ...
4235     Zvi   M      5
4236     Zya   F      8
4237   Zylah   F      5

[4238 rows x 3 columns]
>>> >>> df['istrain'] = np.random.rand(len(df)) < .9
... >>> df
...
        name sex  count  istrain
0      Aaden   M     51     True
1     Aahana   F     26     True
2      Aahil   M      5     True
...      ...  ..    ...      ...
4235     Zvi   M      5     True
4236     Zya   F      8     True
4237   Zylah   F      5     True

[4238 rows x 4 columns]
>>> >>> df.index = pd.MultiIndex.from_tuples(
... ...     zip(df['name'], df['sex']), names=['name_', 'sex_'])
... >>> df
...
               name sex  count  istrain
name_  sex_                            
Aaden  M      Aaden   M     51     True
Aahana F     Aahana   F     26     True
Aahil  M      Aahil   M      5     True
...             ...  ..    ...      ...
Zvi    M        Zvi   M      5     True
Zya    F        Zya   F      8     True
Zylah  F      Zylah   F      5     True

[4238 rows x 4 columns]
>>> >>> df_most_common = {}  # <1>
... >>> for name, group in df.groupby('name'):
... ...     row_dict = group.iloc[group['count'].argmax()].to_dict() # <2>
... ...     df_most_common[(name, row_dict['sex'])] = row_dict
... >>> df_most_common = pd.DataFrame(df_most_common).T  # <3>
...
>>> >>> df_most_common['istest'] = ~df_most_common['istrain'].astype(bool)
... >>> df_most_common
...
            name sex count istrain  istest
Aaden  M   Aaden   M    51    True   False
Aahana F  Aahana   F    26    True   False
Aahil  M   Aahil   M     5    True   False
...          ...  ..   ...     ...     ...
Zvi    M     Zvi   M     5    True   False
Zya    F     Zya   F     8    True   False
Zylah  F   Zylah   F     5    True   False

[4025 rows x 5 columns]
>>> >>> df['istest'] = df_most_common['istest']
... >>> df['istest'] = df['istest'].fillna(False)
... >>> df['istrain'] = ~df['istest']
... >>> istrain = df['istrain']
... >>> df['istrain'].sum() / len(df)
...
0.9091552619159982
>>> >>> unique_names = df['name'][istrain].unique()
... >>> unique_names = df['name'][istrain].unique()
... >>> vectorizer.fit(unique_names)
... >>> vecs = vectorizer.transform(df['name'])
... >>> vecs
...
<4238x3663 sparse matrix of type '<class 'numpy.float64'>'
	with 60542 stored elements in Compressed Sparse Row format>
>>> >>> vecs = pd.DataFrame(vecs.toarray())
... >>> vecs.columns = vectorizer.get_feature_names_out()
... >>> vecs.index = df.index
... >>> vecs.iloc[:,:7]
...
                    A        Aa       Aad       Aah  Aal  Aan  Aar
name_  sex_                                                       
Aaden  M     0.288675  0.288675  0.288675  0.000000  0.0  0.0  0.0
Aahana F     0.218218  0.218218  0.000000  0.218218  0.0  0.0  0.0
Aahil  M     0.288675  0.288675  0.000000  0.288675  0.0  0.0  0.0
...               ...       ...       ...       ...  ...  ...  ...
Zvi    M     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0
Zya    F     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0
Zylah  F     0.000000  0.000000  0.000000  0.000000  0.0  0.0  0.0

[4238 rows x 7 columns]
>>> >>> vectorizer = TfidfVectorizer(analyzer='char',
... ...    ngram_range=(1, 3), use_idf=False, lowercase=False)  # <1>
... >>> vectorizer = vectorizer.fit(unique_names)
... >>> vecs = vectorizer.transform(df['name'])
... >>> vecs = pd.DataFrame(vecs.toarray())
... >>> vecs.columns = vectorizer.get_feature_names_out()
... >>> vecs.index = df.index
... >>> vecs.iloc[:,:5]
...
                    A        Aa       Aad       Aah  Aal
name_  sex_                                             
Aaden  M     0.288675  0.288675  0.288675  0.000000  0.0
Aahana F     0.218218  0.218218  0.000000  0.218218  0.0
Aahil  M     0.288675  0.288675  0.000000  0.288675  0.0
...               ...       ...       ...       ...  ...
Zvi    M     0.000000  0.000000  0.000000  0.000000  0.0
Zya    F     0.000000  0.000000  0.000000  0.000000  0.0
Zylah  F     0.000000  0.000000  0.000000  0.000000  0.0

[4238 rows x 5 columns]
>>> >>> X = vecs.values
... >>> y = (df[['sex']] == 'F').values
... >>> X_train = torch.Tensor(X[istrain])
... >>> X_test = torch.Tensor(X[~istrain])
... >>> y_train = torch.Tensor(y[istrain])
... >>> y_test = torch.Tensor(y[~istrain])
...
>>> >>> import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> range(num_epochs)
range(0, 200)
>>> >>> from tqdm import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> from torch import optim
>>> optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
... optimizer = optim.Adam([var1, var2], lr=0.0001)
...
>>> optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
... optimizer = optim.Adam?
...
>>> optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
... optimizer = optim.Adam(model.parameters(), lr=0.0001)
...
>>> hist -o -p
>>> loss_train
>>> loss_train = torch.nn.BCELoss(weight=torch.Tensor(df[['count']][istrain].values))
>>> loss_train
BCELoss()
>>> >>> from torch.optim import SGD
... >>> hyperparams = {'momentum': 0.001, 'lr': 0.02}  # <1>
... >>> optimizer = SGD(model.parameters(), **hyperparams)  # <2>
... >>> optimizer
...
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.02
    maximize: False
    momentum: 0.001
    nesterov: False
    weight_decay: 0
)
>>> model.parameters()
<generator object Module.parameters at 0x7f4f43a29eb0>
>>> >>> from torch.optim import SGD
... >>> hyperparams = {'momentum': 0.001, 'lr': 0.02}  # <1>
... >>> optimizer = SGD(list(model.parameters()), **hyperparams)  # <2>
... >>> optimizer
...
SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    lr: 0.02
    maximize: False
    momentum: 0.001
    nesterov: False
    weight_decay: 0
)
>>> >>> from tqdm import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> >>> loss_func_train = torch.nn.BCELoss(
... ...     weight=torch.Tensor(df[['count']][istrain].values))
... >>> loss_func_train
...
BCELoss()
>>> >>> from tqdm import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> >>> from tqdm import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> >>> from tqdm import tqdm
... >>> num_epochs = 200
... >>> pbar_epochs = tqdm(range(num_epochs), desc='Epoch:', total=num_epochs)
... 
... >>> for epoch in pbar_epochs:
... ...      optimizer.zero_grad() # <1>
... ...      outputs = model(X_train)
... ...      loss_train = loss_func_train(outputs, y_train) # <2>
... ...      loss_train.backward() # <3>
... ...      optimizer.step() # <4>
...
>>> >>> def make_array(x):
... ...     if hasattr(x, 'detach'):
... ...         return torch.squeeze(x).detach().numpy()
... ...     return x
...
>>> def measure_binary_accuracy(y_pred, y):
...     y_pred = make_array(y_pred).round()
...     y = make_array(y).round()
...     num_correct = (y_pred == y).sum()
...     return num_correct / len(y)
...
>>> >>> def measure_binary_accuracy(y_pred, y):
... ...     y_pred = make_array(y_pred).round()
... ...     y = make_array(y).round()
... ...     num_correct = (y_pred == y).sum()
... ...     return num_correct / len(y)
...
>>> >>> loss_func_test = torch.nn.BCELoss(  # <1>
... ...     weight=torch.Tensor(df[['count']][~istrain].values))
...
>>> for epoch in range(num_epochs):
...     optimizer.zero_grad() 
...     outputs = model(X_train)
...     loss_train = loss_func_train(outputs, y_train) 
...     loss_train.backward() 
...     epoch_loss_train = loss_train.item()
...     optimizer.step() 
...     outputs_test = model(X_test)
...     loss_test = loss_func_test(outputs_test, y_test).item()
...     accuracy_test = measure_binary_accuracy(outputs_test, y_test)
...     if epoch % 20 == 19:
...         print((f'Epoch {epoch}: 
...            loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f},
...            accuracy_test: {accuracy_test:.4f}'))
...
>>> for epoch in range(num_epochs):
...     optimizer.zero_grad() 
...     outputs = model(X_train)
...     loss_train = loss_func_train(outputs, y_train) 
...     loss_train.backward() 
...     epoch_loss_train = loss_train.item()
...     optimizer.step() 
...     outputs_test = model(X_test)
...     loss_test = loss_func_test(outputs_test, y_test).item()
...     accuracy_test = measure_binary_accuracy(outputs_test, y_test)
...     if epoch % 20 == 19:
...         print(f'Epoch {epoch}:'
...             f'loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f},
...             f'accuracy_test: {accuracy_test:.4f}')
...
>>> for epoch in range(num_epochs):
...     optimizer.zero_grad() 
...     outputs = model(X_train)
...     loss_train = loss_func_train(outputs, y_train) 
...     loss_train.backward() 
...     epoch_loss_train = loss_train.item()
...     optimizer.step() 
...     outputs_test = model(X_test)
...     loss_test = loss_func_test(outputs_test, y_test).item()
...     accuracy_test = measure_binary_accuracy(outputs_test, y_test)
...     if epoch % 20 == 19:
...         print(f'Epoch {epoch}:'
...             f' loss_train/test: {loss_train.item():.4f}/{loss_test.item():.4f}'
...             f' accuracy_test: {accuracy_test:.4f}')
...
>>> for epoch in range(num_epochs):
...     optimizer.zero_grad() 
...     outputs = model(X_train)
...     loss_train = loss_func_train(outputs, y_train) 
...     loss_train.backward() 
...     epoch_loss_train = loss_train.item()
...     optimizer.step() 
...     outputs_test = model(X_test)
...     loss_test = loss_func_test(outputs_test, y_test).item()
...     accuracy_test = measure_binary_accuracy(outputs_test, y_test)
...     if epoch % 20 == 19:
...         print(f'Epoch {epoch}:'
...             f' loss_train/test: {loss_train.item():.4f}/{loss_test:.4f},'
...             f' accuracy_test: {accuracy_test:.4f}')
...
>>> vectorizer.transform(['Cason'])
<1x3663 sparse matrix of type '<class 'numpy.float64'>'
	with 12 stored elements in Compressed Sparse Row format>
>>> x_cason = vectorizer.transform(['Cason'])
>>> model(x_cason)
>>> X_train
tensor([[0.2887, 0.2887, 0.2887,  ..., 0.0000, 0.0000, 0.0000],
        [0.2182, 0.2182, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.2887, 0.2887, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]])
>>> model(torch.Tensor(x_cason))
>>> model(torch.Tensor(x_cason.todense()))
tensor([[0.1770]], grad_fn=<SigmoidBackward0>)
>>> x_cason = vectorizer.transform(['John'])
>>> model(torch.Tensor(x_cason.todense()))
tensor([[0.0196]], grad_fn=<SigmoidBackward0>)
>>> x_cason = vectorizer.transform(['Maria'])
>>> model(torch.Tensor(x_cason.todense()))
tensor([[0.8692]], grad_fn=<SigmoidBackward0>)
>>> x_cason = vectorizer.transform(['John', 'Vish', 'Sarah', '])
>>> x_cason = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
>>> x = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
>>> X = vectorizer.transform(['John', 'Vish', 'Sarah', 'Ruby'])
>>> model(torch.Tensor(X.todense()))
tensor([[0.0196],
        [0.6029],
        [0.8199],
        [0.4964]], grad_fn=<SigmoidBackward0>)
>>> hist -o -p
>>> x_vish = vectorizer.transform(['Vish'])
>>> x_vish
<1x3663 sparse matrix of type '<class 'numpy.float64'>'
	with 8 stored elements in Compressed Sparse Row format>
>>> [v for v in x_vish.fillna(0) if v]
>>> [v for v in x_vish.todense().fillna(0) if v]
>>> [v for v in x_vish if v]
>>> [v for v in x_vish[0] if v]
>>> [v for v in x_vish.todense()[0] if v]
>>> x_vish.todense()
matrix([[0., 0., 0., ..., 0., 0., 0.]])
>>> x_vish.todense()[0]
matrix([[0., 0., 0., ..., 0., 0., 0.]])
>>> x_vish.todense()[0,:]
matrix([[0., 0., 0., ..., 0., 0., 0.]])
>>> x_vish.todense().flatten()
matrix([[0., 0., 0., ..., 0., 0., 0.]])
>>> x_vish.todense().ravel()
matrix([[0., 0., 0., ..., 0., 0., 0.]])
>>> x_vish = vectorizer.transform(['Vishvesh'])
>>> model(torch.Tensor(x_vish.todense()))
tensor([[0.3729]], grad_fn=<SigmoidBackward0>)
>>> >>> X = vectorizer.transform(
... ...     ['John', 'Vishvesh', 'Greg',  # <1>
... ...     'Sarah', 'Ruby'])  # <2>
... >>> model(torch.Tensor(X.todense()))
...
tensor([[0.0196],
        [0.3729],
        [0.1808],
        [0.8199],
        [0.4964]], grad_fn=<SigmoidBackward0>)
>>> >>> X = vectorizer.transform(
... ...     ['John', 'Greg', 'Vishvesh',  # <1>
... ...      'Sarah', 'Ruby', 'Carlana'])  # <2>
... >>> model(torch.Tensor(X.todense()))
...
tensor([[0.0196],
        [0.1808],
        [0.3729],
        [0.8199],
        [0.4964],
        [0.8062]], grad_fn=<SigmoidBackward0>)
>>> >>> X = vectorizer.transform(
... ...     ['John', 'Greg', 'Vishvesh',  # <1>
... ...      'Ruby', 'Sarah', 'Carlana'])  # <2>
... >>> model(torch.Tensor(X.todense()))
...
tensor([[0.0196],
        [0.1808],
        [0.3729],
        [0.4964],
        [0.8199],
        [0.8062]], grad_fn=<SigmoidBackward0>)
>>> >>> X = vectorizer.transform(
... ...     ['John', 'Greg', 'Vishvesh',  # <1>
... ...      'Ruby', 'Carlana', 'Sarah'])  # <2>
... >>> model(torch.Tensor(X.todense()))
...
tensor([[0.0196],
        [0.1808],
        [0.3729],
        [0.4964],
        [0.8062],
        [0.8199]], grad_fn=<SigmoidBackward0>)
>>> hist -o -p -f ~/code/tangibleai/nlpia2/src/nlpia2/ch05/single_neuron_sex.hist.ipy.md
