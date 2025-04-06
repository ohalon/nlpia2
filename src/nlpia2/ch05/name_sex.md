You can find US Census data on Data World.

Baby names by State and Territory are here: https://data.world/ssa/baby-names-for-us-states-territories

We then need to preprocess the data before we use it for modeling.

- downsample to reduce the memory required
- split into training and test sets
- tfidf vectors


```python

>>> import pandas as pd
>>> pd.read_csv('/home/hobs/Downloads/baby-names-state.csv')
        state_abb sex  year      name  count
0              AK   F  1910      Mary     14
1              AK   F  1910     Annie     12
2              AK   F  1910      Anna     10
3              AK   F  1910  Margaret      8
4              AK   F  1910     Helen      7
...           ...  ..   ...       ...    ...
6215829        WY   M  2020     Simon      5
6215830        WY   M  2020  Sterling      5
6215831        WY   M  2020   Stetson      5
6215832        WY   M  2020   Timothy      5
6215833        WY   M  2020    Wesley      5

[6215834 rows x 5 columns]
>>> df = _
>>> df.to_csv('/home/hobs/.nlpia2-data/baby-names-state.csv.gz', index=False)
>>> df.describe(include='all')
       state_abb      sex          year     name         count
count    6215834  6215834  6.215834e+06  6215834  6.215834e+06
unique        51        2           NaN    32110           NaN
top           CA        F           NaN   Leslie           NaN
freq      394179  3459588           NaN     7378           NaN
mean         NaN      NaN  1.976508e+03      NaN  5.097924e+01
std          NaN      NaN  3.104012e+01      NaN  1.741975e+02
min          NaN      NaN  1.910000e+03      NaN  5.000000e+00
25%          NaN      NaN  1.953000e+03      NaN  7.000000e+00
50%          NaN      NaN  1.982000e+03      NaN  1.200000e+01
75%          NaN      NaN  2.004000e+03      NaN  3.300000e+01
max          NaN      NaN  2.020000e+03      NaN  1.002500e+04

>>> pd.read_csv('/home/hobs/Downloads/baby-names-territories.csv')
      territory_code sex  year      name  count
0                 PR   F  1998     Paola    724
1                 PR   F  1998   Genesis    500
2                 PR   F  1998  Gabriela    447
3                 PR   F  1998    Nicole    392
4                 PR   F  1998   Alondra    344
...              ...  ..   ...       ...    ...
25534             TR   M  2020  Jeremiah      6
25535             TR   M  2020      Liam      6
25536             TR   M  2020     Elias      5
25537             TR   M  2020     Henry      5
25538             TR   M  2020      Noah      5

[25539 rows x 5 columns]
>>> dft = _
>>> dft['state_abb'] = dft['territory_code']
>>> del dft['territory_code']
>>> dft.head()
  sex  year      name  count state_abb
0   F  1998     Paola    724        PR
1   F  1998   Genesis    500        PR
2   F  1998  Gabriela    447        PR
3   F  1998    Nicole    392        PR
4   F  1998   Alondra    344        PR
>>> df.head()
  state_abb sex  year      name  count
0        AK   F  1910      Mary     14
1        AK   F  1910     Annie     12
2        AK   F  1910      Anna     10
3        AK   F  1910  Margaret      8
4        AK   F  1910     Helen      7
>>> df = pd.concat([df, dft])
>>> df.head()
  state_abb sex  year      name  count
0        AK   F  1910      Mary     14
1        AK   F  1910     Annie     12
2        AK   F  1910      Anna     10
3        AK   F  1910  Margaret      8
4        AK   F  1910     Helen      7
>>> df.describe(include='all')
       state_abb      sex          year     name         count
count    6241373  6241373  6.241373e+06  6241373  6.241373e+06
unique        53        2           NaN    33203           NaN
top           CA        F           NaN    James           NaN
freq      394179  3472023           NaN     7399           NaN
mean         NaN      NaN  1.976636e+03      NaN  5.087657e+01
std          NaN      NaN  3.104305e+01      NaN  1.738913e+02
min          NaN      NaN  1.910000e+03      NaN  5.000000e+00
25%          NaN      NaN  1.953000e+03      NaN  7.000000e+00
50%          NaN      NaN  1.982000e+03      NaN  1.200000e+01
75%          NaN      NaN  2.004000e+03      NaN  3.300000e+01
max          NaN      NaN  2.020000e+03      NaN  1.002500e+04
>>> dft.describe(include='all')
          sex          year   name         count state_abb
count   25539  25539.000000  25539  25539.000000     25539
unique      2           NaN   3210           NaN         2
top         M           NaN  Angel           NaN        PR
freq    13104           NaN     48           NaN     22477
mean      NaN   2007.692823    NaN     25.887897       NaN
std       NaN      6.288080    NaN     60.528368       NaN
min       NaN   1998.000000    NaN      5.000000       NaN
25%       NaN   2002.000000    NaN      6.000000       NaN
50%       NaN   2007.000000    NaN      9.000000       NaN
75%       NaN   2013.000000    NaN     19.000000       NaN
max       NaN   2020.000000    NaN   1472.000000       NaN
>>> df['name'].apply(lambda s: ' ' in s).sum()
0

>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))

>>> from sklearn.linear_model import Lasso, LogisticRegression
>>> df.columns
Index(['state_abb', 'sex', 'year', 'name', 'count'], dtype='object')
>>> df.columns = ['region'] + list(df.columns)[1:]
>>> df.head()
  region sex  year      name  count
0     AK   F  1910      Mary     14
1     AK   F  1910     Annie     12
2     AK   F  1910      Anna     10
3     AK   F  1910  Margaret      8
4     AK   F  1910     Helen      7
>>> df.to_csv('/home/hobs/.nlpia2-data/baby-names-region.csv.gz', index=False)
>>> df = df.sample(1000000)
>>> df.head()
        region sex  year      name  count
749805      CA   M  2008   Domenik      5
5270876     TX   F  1954   Unknown     16
6174014     WV   M  1976     Tracy     16
4187958     NY   M  1985  Menachem     38
5776190     VA   M  1942   Percell     10

# You can't change a group and expect it to persist once the groupby exits
>>> for y, g in df.groupby('year'):
...     tot = g['count'].sum()
...     print(y, tot)
...     df[y == df['year']]['count'] = -1
...
>>> df.head()
        region sex  year      name  count
749805      CA   M  2008   Domenik      5
5270876     TX   F  1954   Unknown     16
6174014     WV   M  1976     Tracy     16
4187958     NY   M  1985  Menachem     38
5776190     VA   M  1942   Percell     10
>>> for y in df['year'].unique():
...     g = df[y == df['year']]
...     tot = g['count'].sum()
...     print(y, tot)
...     df[y == df['year']]['count'] *= -1
...
>>> df.head()
        region sex  year      name  count
749805      CA   M  2008   Domenik      5
5270876     TX   F  1954   Unknown     16
6174014     WV   M  1976     Tracy     16
4187958     NY   M  1985  Menachem     38
5776190     VA   M  1942   Percell     10
>>> for y in df['year'].unique():
...     g = df[y == df['year']]
...     tot = g['count'].sum()
...     
...     mask = y == df['year']
...     df['count'][mask] = df['count'][mask].copy() / df['count'][mask].sum()
...     print(df[mask].sum())
>>> df.head()
        region sex  year      name     count
749805      CA   M  2008   Domenik  0.000009
5270876     TX   F  1954   Unknown  0.000028
6174014     WV   M  1976     Tracy  0.000034
4187958     NY   M  1985  Menachem  0.000069
5776190     VA   M  1942   Percell  0.000026
>>> df.columns = list(df.columns)[:-1] + ['frequency']

>>> vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
>>> vectorizer.fit(df['name'])
TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
>>> vecs = vectorizer.transform(df['name'])
>>> df.columns = list(df.columns)[:-1] + ['freq']
>>> model.fit(vecs[istrain], df['sex'][istrain], sample_weight=df['freq'][istrain])
LogisticRegression()
>>> model.score(vecs[istrain], df['sex'][istrain], sample_weight=df['freq'][istrain])
0.812830747076089
>>> model.score(vecs[~istrain], df['sex'][~istrain], sample_weight=df['freq'][~istrain])
0.8082910347803947
>>> names = [
...    'Maria', 'Aditi', 'Jessica', 'Olessya', 'Una', 'Hanna',
...    'Winnie', 'Olessya', 'Sylvia', 
...    'Vish', 'Mohammed', 'Jon', 'John', 'Ted', 'Kazuma', 'Meijke'
...    ]
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
Kazuma      F
Meijke      M
dtype: object
```
