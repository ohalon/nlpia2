""" Notes from mob programming session Wed Oct 27, 2021 5:30 Pacific
(email engineering@tangibleai.com for details)

You need to have nlpia2/src/nlpia2 in your python path (or your CWD)
  for this to work

See [README.md](https://gitlab.com/prosocialai/nlpia2/README.md) for details.

You can `pip install nlpia2`
OR
`git clone git@gitlab.com:prosocialai/nlpia2`
then `cd nlpia2/src/nlpia2`
"""

"""
>>> from tqdm import tqdm
>>> import pandas as pd
>>> import numpy as np
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from sklearn.linear_model import LogisticRegression
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # noqa


"""
>>> from nlpia2.constants import DATA_DIR
>>> DATA_DIR
PosixPath('.../.nlpia-data/')
"""
from nlpia2.constants import DATA_DIR


"""
>>> np.random.seed(451)
>>> df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')
>>> df.sample(3).T
         6139665   2565339   22297
region        WV        MD        AK
sex            F         F         M
year        1987      1954      1988
name    Brittani       Ida   Maxwell
count         10        18         5
freq    0.000003  0.000005  0.000001
"""
df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')
df.sample(3).T
#          6139665   2565339   22297
# region        WV        MD        AK
# sex            F         F         M
# year        1987      1954      1988
# name    Brittani       Ida   Maxwell
# count         10        18         5
# freq    0.000003  0.000005  0.000001


"""
>>> istrain = np.random.rand(len(df))
>>> istrain.sum() / len(istrain)
0.8997952854283825
"""
np.random.seed(451)
istrain = np.random.rand(len(df)) < .9
istrain.sum() / len(istrain)
# 0.8997952854283825

"""
>>> vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
>>> vectorizer.fit(tqdm(df['name'][istrain]))
>>> feature_names = list(vectorizer.get_feature_names_out())
>>> len(feature_names)
5964
>>> feature_names[:5]
['a', 'aa', 'aab', 'aac', 'aad']
"""
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(tqdm(df['name'][istrain]))
feature_names = list(vectorizer.get_feature_names_out())
len(feature_names)
# 5964
feature_names[:5]
# ['a', 'aa', 'aab', 'aac', 'aad']


"""
>>> vecs = vectorizer.transform(tqdm(df['name']))
>>> vecs
<6241373x5964 sparse matrix of type '<class 'numpy.float64'>'
    with 86718620 stored elements in Compressed Sparse Row format>
"""
vecs = vectorizer.transform(tqdm(df['name']))
vecs
# <6241373x5964 sparse matrix of type '<class 'numpy.float64'>'
#     with 86718620 stored elements in Compressed Sparse Row format>


"""
>>> model = LogisticRegression(C=1, max_iter=2000)
>>> model.fit(vecs[istrain], df['sex'][istrain], sample_weight=df['freq'][istrain])
LogisticRegression(C=1, max_iter=2000)
"""
model = LogisticRegression(C=1, max_iter=2000)
model.fit(vecs[istrain], df['sex'][istrain], sample_weight=df['freq'][istrain])
# LogisticRegression(max_iter=2000)


"""
>>> names = [
...     'Maria', 'Aditi', 'Jessica', 'Olessya', 'Una',
...     'Hanna', 'Winnie', 'Olessya', 'Sylvia',
...     'Vish', 'Mohammed', 'Jon', 'John', 'Ted',
...     'Meijke', 'Uzo', 'Kazuma', 'Kemal']
>>> pd.Series(model.predict(vectorizer.transform(names)), index=names)
Maria       F
Aditi       F
Jessica     F
Olessya     F
Una         F
Hanna       F
Winnie      F
Olessya     F
Sylvia      F
Vish        M
Mohammed    M
Jon         M
John        M
Ted         M
Meijke      M
Uzo         M
Kazuma      F
Kemal       F
"""
names = [
    'Maria', 'Aditi', 'Jessica', 'Olessya', 'Una',
    'Hanna', 'Winnie', 'Olessya', 'Sylvia',
    'Vish', 'Mohammed', 'Jon', 'John', 'Ted',
    'Meijke', 'Uzo', 'Kazuma', 'Kemal']
print(pd.Series(model.predict(vectorizer.transform(names)),
                index=names))
# Maria       F
# Aditi       F
# Jessica     F
# Olessya     F
# Una         F
# Hanna       F
# Winnie      F
# Olessya     F
# Sylvia      F
# Vish        M
# Mohammed    M
# Jon         M
# John        M
# Ted         M
# Meijke      M
# Uzo         M
# Kazuma      F
# Kemal       F
# dtype: object


"""
>>> vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3),
...                              lowercase=False)
>>> vectorizer.fit(tqdm(df['name'][istrain]))
TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(1, 3))
"""
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3),
                             lowercase=False)
print(vectorizer.fit(tqdm(df['name'][istrain])))
# TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(1, 3))


"""
>>> feature_names = list(vectorizer.get_feature_names_out())
>>> len(feature_names)
8624
>>> feature_names[:5]
['A', 'Aa', 'Aab', 'Aad', 'Aah']
"""
feature_names = list(vectorizer.get_feature_names_out())
print(len(feature_names))
# 8624
print(feature_names[:5])
# ['A', 'Aa', 'Aab', 'Aad', 'Aah']


"""
>>> vecs = vectorizer.transform(tqdm(df['name']))
>>> vecs
<6241373x8624 sparse matrix of type '<class 'numpy.float64'>'
    with 87648303 stored elements in Compressed Sparse Row format>
"""
vecs = vectorizer.transform(tqdm(df['name']))
vecs
# <6241373x8624 sparse matrix of type '<class 'numpy.float64'>'
#     with 87648303 stored elements in Compressed Sparse Row format>


"""
>>> model = LogisticRegression(C=1, max_iter=2000)
>>> model.fit(vecs[istrain], df['sex'][istrain],
...           sample_weight=df['freq'][istrain])
LogisticRegression(C=1, max_iter=2000)
"""
model = LogisticRegression(C=1, max_iter=2000)
model.fit(vecs[istrain], df['sex'][istrain],
          sample_weight=df['freq'][istrain])
# LogisticRegression(C=1, max_iter=2000)


"""
>>> names = [
...     'Maria', 'Aditi', 'Jessica', 'Una',
...     'Hanna', 'Winnie', 'Olessya', 'Sylvia',
...     'Vish', 'Mohammed', 'Jon', 'John', 'Ted',
...     'Meijke', 'Uzo', 'Kazuma', 'Kemal']
>>> pd.Series(model.predict(vectorizer.transform(names)), index=names)
Maria       F
Aditi       F
Jessica     F
Una         F
Hanna       F
Winnie      F
Olessya     M
Sylvia      F
Vish        M
Mohammed    M
Jon         M
John        M
Ted         M
Meijke      M
Uzo         M
Kazuma      F
Kemal       M
"""
names = [
    'Maria', 'Aditi', 'Jessica', 'Una',
    'Hanna', 'Winnie', 'Olessya', 'Sylvia',
    'Vish', 'Mohammed', 'Jon', 'John', 'Ted',
    'Meijke', 'Uzo', 'Kazuma', 'Kemal']
print(pd.Series(model.predict(vectorizer.transform(names)),
                index=names))


"""
>>> (df['name'] == 'Kazuma').sum()
0
>>> (df['name'] == 'Kaz').sum()
1
>>> df[df['name'] == 'Kaz']
        region sex  year name  count      freq
2424961     LA   M  2015  Kaz      5  0.000002
"""
(df['name'] == 'Kazuma').sum()
0
(df['name'] == 'Kaz').sum()
1
df[df['name'] == 'Kaz']
#         region sex  year name  count      freq
# 2424961     LA   M  2015  Kaz      5  0.000002


"""
>>> (df['name'].str.startswith('Kaz')).sum()
151
>>> iskaz = df['name'].str.startswith('Kaz')
>>> df[iskaz].sort_values('freq')
        region sex  year      name  count      freq
4225691     NY   M  2006      Kazi      5  0.000001
4225692     NY   M  2006    Kazuki      5  0.000001
562866      CA   F  2005  Kazandra      5  0.000001
4223685     NY   M  2005      Kazi      5  0.000001
5360426     TX   F  2000  Kazandra      5  0.000001
...        ...  ..   ...       ...    ...       ...
625232      CA   M  1918     Kazuo     22  0.000011
626674      CA   M  1921     Kazuo     25  0.000012
623394      CA   M  1913     Kazuo     13  0.000013
627194      CA   M  1922     Kazuo     28  0.000013
399285      CA   F  1927    Kazuko     31  0.000014
[151 rows x 6 columns]
"""


"""
>>> model.coef_
array([[-5.56584447e-01, 5.14287250e-02, 2.69276390e-06, ...,
        -1.39283459e-04, -1.21600561e-04, -9.11513939e-05]])
>>> coef = pd.Series(model.coef_[0], index=feature_names)
>>> coef
A     -0.143835
Aa     0.033721
Aab    0.000001
Aad    0.000214
Aah    0.000008
         ...
zze   -0.000098
zzi   -0.001747
zzl   -0.000139
zzm   -0.000120
zzy   -0.000094
Length: 8624, dtype: float64
>>> coef.sort_values().round(2)
a     -0.50
i     -0.34
Mar   -0.30
y     -0.27
Ma    -0.25
       ...
mes    0.27
cha    0.27
J      0.28
am     0.32
o      0.40
"""
coef = pd.Series(model.coef_[0], index=feature_names)
print(coef.sort_values().round(2))
# a     -0.50
# i     -0.34
# Mar   -0.30
# y     -0.27
# Ma    -0.25
#        ...
# mes    0.27
# cha    0.27
# J      0.28
# am     0.32
# o      0.40


"""
>>> kazvec = vectorizer.transform(['Kazuma'])
>>> kazvec
<1x8624 sparse matrix of type '<class 'numpy.float64'>'
    with 14 stored elements in Compressed Sparse Row format>
>>> kazvec = pd.Series(kazvec.toarray()[0], index=feature_names)
>>> kazvec = kazvec[kazvec.abs() > 0]
>>> kazvec
K      0.126702
Ka     0.154755
Kaz    0.390671
a      0.100628
az     0.235593
azu    0.398854
m      0.115413
ma     0.159887
u      0.114679
um     0.248047
uma    0.312667
z      0.175519
zu     0.341750
zum    0.473007
"""
# TFIDFVectorizer.transform creates a sparse matrix, 1 row per input str
kazvec = vectorizer.transform(['Kazuma'])
print(kazvec)
# <1x8624 sparse matrix of type '<class 'numpy.float64'>'
#     with 14 stored elements in Compressed Sparse Row format>
pd.Series(kazvec.toarray()[0], index=feature_names)
kazvec = pd.Series(kazvec.toarray()[0], index=feature_names)
kazvec = kazvec[kazvec.abs() > 0]
kazvec
# K      0.126702
# Ka     0.154755
# Kaz    0.390671
# a      0.100628
# az     0.235593
# azu    0.398854
# m      0.115413
# ma     0.159887
# u      0.114679
# um     0.248047
# uma    0.312667
# z      0.175519
# zu     0.341750
# zum    0.473007

"""
>>> np.array(kazvec[kazvec != 0])[0]
array([0.10064136, 0.24160243, 0.36640437, 0.11589853, 0.1563424,
       0.40920386, 0.09844366, 0.12768782, 0.12058102, 0.26053805,
       0.32475817, 0.17116995, 0.31639993, 0.49863121])
>>> coef
a - 0.556584
aa     0.051429
aab    0.000003
aac    0.021992
aad    0.000415
    ...
zze - 0.000103
zzi - 0.001919
zzl - 0.000139
zzm - 0.000122
zzy - 0.000091
Length: 5986, dtype: float64
>>> coef.values[np.array(kazvec[kazvec != 0])[0] > 0]
>>> coef.values[kazvec != 0]
>>> type(kazvec != 0)
scipy.sparse.csr.csr_matrix
>>> (kazvec != 0).todense()
matrix([[True, False, False, ..., False, False, False]])
>>> np.array((kazvec != 0).todense())[0]
array([True, False, False, ..., False, False, False])
>>> coef[np.array((kazvec != 0).todense())[0] != 0]
a - 0.556584
az - 0.022166
azu - 0.000366
k      0.066347
ka - 0.182166
kaz    0.000030
m - 0.031816
ma - 0.280481
u      0.029682
um - 0.008430
uma    0.000764
z - 0.066873
zu - 0.001613
zum    0.000006
dtype: float64
>>> model.predict(kazvec)
array(['F'], dtype=object)
>>> model.predict_proba(kazvec)
array([[0.51219282, 0.48780718]])
>>> pd.Series(model.predict_proba(vectorizer.transform(names))[:, 1], index=names)
Maria       0.381240
Aditi       0.477269
Jessica     0.483249
Olessya     0.499430
Una         0.481148
Hanna       0.426783
Winnie      0.484074
Olessya     0.499430
Sylvia      0.484732
Vish        0.516631
Mohammed    0.569390
Jon         0.619116
John        0.678048
Ted         0.546004
Kazuma      0.487807
Meijke      0.528277
Kemal       0.494326
dtype: float64
>>> model.classes_
array(['F', 'M'], dtype=object)
>>> hist
>>> hist - o - p - f kemal_kazuma_labeled_female.md
"""

"""
>>> import yaml
>>> women = yaml.full_load(
...     str(DATA_DIR / 'women-in-ai-ethics-and-prosocial-ai.yml'))
>>> list(women)[:3]
['Vinita Silaparasetty', 'Ivana Bartoletti', 'Karolyn Gainfort']
>>> women = [w.split() for w in women]
>>> women = [w[1] if w[0] == 'Dr.' else w[0] for w in women]
>>> women[:6]
['Vinita', 'Ivana', 'Karolyn', 'Bulbul', 'Moojan', 'Hanan']
>>> pd.Series(model.predict(vectorizer.transform(women)), index=women)
Vinita     F
Ivana      F
Karolyn    F
Bulbul     M
Moojan     M
Hanan      F
Nikita     F
Zeynep     F
Melanie    F
Kathryn    F
Helen      F
Susan      F
Joy        M
Maria      F
"""
