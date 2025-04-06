import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.constants import DATA_DIR

df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')


""" Add a freq column based on the counts, to use as sample_weights in LogisticRegression.fit

>>> df['freq'] = 0.
>>> for y in df['year'].unique():
...     g = df[y == df['year']]
...     tot = g['count'].sum()
...     mask = y == df['year']
...     df['freq'][mask] = df['count'][mask] / df['count'][mask].values.sum()
...     print(y, tot, df['freq'][mask].sum())

>>> df['freq'].sum().round(10)
111.0
>>> df['year'].nunique()
111
"""

df = pd.read_csv('.nlpia2-data/baby-names-region.csv.gz')
df['freq'] = 0.
for y in df['year'].unique():
    g = df[y == df['year']]
    tot = g['count'].sum()
    mask = y == df['year']
    df['freq'][mask] = df['count'][mask] / df['count'][mask].values.sum()
    print(y, tot, df['freq'][mask].sum())
df['freq'].sum().round(10)
df['year'].nunique()
df.to_csv('.nlpia2-data/baby-names-region.csv.gz', index=False)

# verify minimial loss of precision during small value sum:
pd.read_csv('.nlpia2-data/baby-names-region.csv.gz')
for y in df['year'].unique():
    g = df[y == df['year']]
    tot = g['count'].sum()
    mask = y == df['year']
    print(y, tot, df['freq'][mask].sum())


df = pd.read_csv('/home/hobs/.nlpia2-data/baby-names-region.csv.gz')

# df = df.sample(1_000_000, random_state=1989)
np.random.seed(451)
istrain = np.random.rand(len(df)) < .9

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(df['name'][istrain])
vecs = vectorizer.transform(df['name'])
