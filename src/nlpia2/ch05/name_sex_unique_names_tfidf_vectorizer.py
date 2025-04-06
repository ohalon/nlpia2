"""
>>> model = LogisticRegression()
>>> X = df['name'].str.len()
>>> df['len'] = df['name'].str.len()
>>> X = df[['len']]
>>> y = df['sex']
"""

"""
>>> model.fit(X[istrain], y[istrain])
LogisticRegression()
>>> model.score(X[istrain], y[istrain])
0.560972925379705
>>> model.score(X[~istrain], y[~istrain])
0.5492341356673961
"""

"""
>>> model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
LogisticRegression()

>>> model.score(X[istrain], y[istrain])
0.5173893902707462
>>> model.score(X[~istrain], y[~istrain])
0.5361050328227571

>>> model.score(X[istrain], y[istrain], sample_weight=df['count'][istrain])
0.5511512294352704
>>> model.score(X[~istrain], y[~istrain], sample_weight=df['count'][~istrain])
0.5972233131725709
"""

# neither year nor len are statistically significant predictors of sex
from pathlib import Path
import numpy as np
import pandas as pd
DATA_DIR = Path('.nlpia2-data')
df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')

np.random.seed(451)
df = df.sample(10_000)
names = df['name'].unique()
istrain = np.random.rand(len(names)) < .9
istest = ~istrain
df.groupby('name')['istrain'] = istrain
groups = df.groupby('name')
dfistrain = pd.DataFrame([istrain]).T
dfistrain
dfistrain.columns = ['istrain']
sum(dfistrain)
dfistrain.sum()
dfistrain.index = names
df.merge(dfistrain, on='name', kind='left')
df.merge?
df.merge?
df.merge(dfistrain, on='name')
dfistrain?
dfistrain
dfistrain['name'] = names
df.merge(dfistrain, on='name')
hist

istrain = df.merge(dfistrain, on='name')['istrain']
istrain
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(df['name'].str.len()[istrain], df['sex'])
istrain = istrain.values
model.fit(df['name'].str.len()[istrain], df['sex'])
model.fit(df[['name']].str.len()[istrain], df['sex'])
X = df['name'].str.len()
df['len'] = df['name'].str.len()
X = df[['len']]
y = df['sex']
model.fit(X[istrain], y[istrain])
mmodel.score(X[istrain], y[istrain])
model.score(X[istrain], y[istrain])
model.score(X[~istrain], y[~istrain])
history -o -p
model.score(X[istrain], y[istrain], sample_weight=df['count'])
model.score(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[istrain], y[istrain])
model.score(X[~istrain], y~[istrain])
model.score(X[~istrain], y[~istrain])
model.score(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[~istrain], y[~istrain], sample_weight=df['count'][~istrain])


