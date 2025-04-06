import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression, Lasso
from nlpia.constants import DATA_DIR

df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')
names = df.set_index('name').groupby('name')['count'].sum()


# df = df.sample(1_000_000, random_state=1989)
np.random.seed(451)
istrain = np.random.rand(len(names)) < .9

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(names.index.values[istrain])
vecs = vectorizer.transform(names.index)


"""
>>> from nlpia.constants import DATA_DIR
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(451)

>>> df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')  # <1>
>>> df.sample(3).T
         6139665   2565339   22297
region        WV        MD        AK
sex            F         F         M
year        1987      1954      1988
name    Brittani       Ida   Maxwell
count         10        18         5
freq    0.000003  0.000005  0.000001
"""

"""
>>> df = df.set_index(['name', 'sex'])
>>> namesex = df.groupby(['name', 'sex'])
>>> counts = namesex['count'].sum()
>>> counts
name      sex
Aaban     M        12
Aadam     M         6
Aadan     M        23
Aadarsh   M        11
Aaden     M      4172
                 ...
Zyren     M         6
Zyria     F        81
Zyriah    F        63
Zyron     M         5
Zyshonne  M         5
Name: count, Length: 36343, dtype: int64
"""

"""
>>> counts[('Dewey', 'M')]
26806
>>> counts[('Dewey', 'F')]
5
>>> counts[('John', 'F')]
15676
>>> counts[('John', 'M')]
4890043
>>> counts[('Hobson', 'M')]
33
>>> counts[('Hobson', 'F')]
KeyError: ('Hobson', 'F')
>>> counts[('Maria', 'F')]
542990
>>> counts[('Maria', 'M')]
2730

>>> counts[('Carlana', 'F')]
KeyError
>>> counts[('Carlana', 'M')]
KeyError
>>> counts[('Cason', 'M')]
8346
>>> counts[('Cason', 'F')]
KeyError
>>> counts[('Robin', 'F')]
287514
>>> counts[('Robin', 'M')]
40727
>>> counts[('Clayton', 'M')]
128929
>>> counts[('Clayton', 'F')]
19
>>> df.groupby(['region','year'])['count'].sum()
region  year
AK      1910     115
        1911      84
        1912     141
        1913     110
        1914     245
                ...
WY      2016    2888
        2017    2578
        2018    2391
        2019    2377
        2020    2075
Name: count, Length: 5707, dtype: int64
>>> msyr = _
>>> msyr = msyr.loc[('MS', )]
>>> msyr
year
1910    16902
1911    14831
1912    21058
1913    22603
1914    27021
        ...
2016    24477
2017    24187
2018    23624
2019    23168
2020    21941
Name: count, Length: 111, dtype: int64
>>> msyr.sum()
4138235
"""

"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> np.random.seed(451)
>>> istrain = np.random.rand(len(names)) < .9
"""

"""
>>> vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
>>> vectorizer.fit(names.index.values[istrain])
>>> vecs = vectorizer.transform(names.index)
>>> vecs
<33203x5898 sparse matrix of type '<class 'numpy.float64'>'
    with 501031 stored elements in Compressed Sparse Row format>
"""


""" The `vectorizer.use_idf=True`

The `vectorizer.use_idf=True` argument means that the
rows of the TF-IDF matrix have been normalized to have unit length (L2 norm).

>>> np.linalg.norm(vecs[0].toarray()[0])
0.9999999999999999
"""

"""
>>> names
name
Aaban         12
Aadam          6
Aadan         23
Aadarsh       11
Aaden       4172
            ...
Zyren          6
Zyria         81
Zyriah        63
Zyron          5
Zyshonne       5
Name: count, Length: 33203, dtype: int64
"""

"""
>>> df = pd.DataFrame(
...     [list(tup) for tup in names.index.values],
...     columns=['name', 'sex']
...     )
"""

""" Normalize by the original counts of birth certificates
>>> vecs
<33203x5898 sparse matrix of type '<class 'numpy.float64'>'
    with 501031 stored elements in Compressed Sparse Row format>
>>> vecs /= names.values.reshape(33203, 1).dot(np.ones((1,vecs.shape[1])))
>>> np.linalg.norm(vecs[0].toarray()[0])
>>> np.array(vecs[0])[0]
array([0.01742584, 0.02463093, 0.0450051 , ..., 0.        , 0.        ,
       0.        ])
>>> np.linalg.norm(np.array(vecs[0])[0])
0.08333333333333334
>>> vecs.div(vecs.sum(axis=1), axis=0)
>>> pd.DataFrame(vecs).div(vecs.sum(axis=1), axis=0)
           0         1         2     3         4     5     6     ...  5891  5892  5893  5894  5895  5896  5897
0      0.072670  0.102717  0.187681   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
1      0.081247  0.114840  0.000000   0.0  0.165448   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
2      0.086992  0.122961  0.000000   0.0  0.177147   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
3      0.052602  0.074351  0.000000   0.0  0.107117   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
4      0.056743  0.120308  0.000000   0.0  0.173325   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
...         ...       ...       ...   ...       ...   ...   ...  ...   ...   ...   ...   ...   ...   ...   ...
33198  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33199  0.025296  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33200  0.020402  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33201  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33202  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0

[33203 rows x 5898 columns]
>>> df = _
>>> df.columns = vectorizer.get_feature_names_out()
>>> df.head()
          a        aa       aab  aac       aad  aaf  aah  aai  aaj  ...  zyr  zys   zz  zza  zze  zzi  zzl  zzm  zzy
0  0.072670  0.102717  0.187681  0.0  0.000000  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
1  0.081247  0.114840  0.000000  0.0  0.165448  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
2  0.086992  0.122961  0.000000  0.0  0.177147  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
3  0.052602  0.074351  0.000000  0.0  0.107117  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
4  0.056743  0.120308  0.000000  0.0  0.173325  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

[5 rows x 5898 columns]
>>> np.linalg.norm(df.iloc[0])
0.3475188196001824
>>> pd.DataFrame(vecs).div(np.linalg.norm(vecs, axis=1), axis=0)
           0         1         2     3         4     5     6     ...  5891  5892  5893  5894  5895  5896  5897
0      0.209110  0.295571  0.540061   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
1      0.242669  0.343005  0.000000   0.0  0.494161   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
2      0.255215  0.360738  0.000000   0.0  0.519709   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
3      0.195023  0.275659  0.000000   0.0  0.397137   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
4      0.170997  0.362550  0.000000   0.0  0.522318   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
...         ...       ...       ...   ...       ...   ...   ...  ...   ...   ...   ...   ...   ...   ...   ...
33198  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33199  0.076905  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33200  0.070272  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33201  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0
33202  0.000000  0.000000  0.000000   0.0  0.000000   0.0   0.0  ...   0.0   0.0   0.0   0.0   0.0   0.0   0.0

[33203 rows x 5898 columns]
>>> df = _
>>> np.linalg.norm(df.iloc[0])
1.0
>>> hist -o -p
"""

"""
>>> df = df.set_index(['name', 'sex'])
>>> namesex = df.groupby(['name', 'sex'])
>>> counts = namesex['count'].sum()
>>> counts
name      sex
Aaban     M        12
Aadam     M         6
Aadan     M        23
Aadarsh   M        11
Aaden     M      4172
                 ...
Zyren     M         6
Zyria     F        81
Zyriah    F        63
Zyron     M         5
Zyshonne  M         5
"""

"""
>>> counts[('Maria',)]
sex
F    542990
M      2730
>>> counts[('Avi',)]
sex
F      84
M    2981
Name: count, dtype: int64
"""

"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(analyzer='char',
...    ngram_range=(1, 3), use_idf=False)  # <1>

<1> `use_idf=False` prevents the vectorizer from using
inverse document frequency to normalize the vectors.
You will do that manually with the birth counts.
"""

"""
>>> df = pd.DataFrame([list(tup) for tup in names.index.values],
...                   columns=['name', 'sex'])
>>> df
           name sex
0         Aaban   M
1         Aadam   M
2         Aadan   M
3       Aadarsh   M
4         Aaden   M
...         ...  ..
36338     Zyren   M
36339     Zyria   F
36340    Zyriah   F
36341     Zyron   M
36342  Zyshonne   M
"""
