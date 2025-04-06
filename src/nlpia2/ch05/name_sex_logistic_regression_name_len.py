# neither year nor len are statistically significant predictors of sex
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # , Lasso

DATA_DIR = Path('.nlpia2-data')
df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')

np.random.seed(451)
df = df.sample(10_000)
names = df['name'].unique()
istrain = np.random.rand(len(names)) < .9
istest = ~istrain

dfistrain = pd.DataFrame([istrain]).T
dfistrain.columns = ['istrain']
dfistrain.sum()
dfistrain['name'] = names
istrain = df.merge(dfistrain, on='name')['istrain']


# istrain = np.random.rand(len(df)) < .9
df['len'] = df['name'].str.len()
model = LogisticRegression(class_weight='balanced', max_iter=2000)
model.fit(df[['len', 'year']][istrain], df['sex'][istrain], sample_weight=df['count'][istrain])
model.score(df[['len', 'year']][istrain], df['sex'][istrain], sample_weight=df['count'][istrain])
model.score(df[['len', 'year']][~istrain], df['sex'][~istrain], sample_weight=df['count'][~istrain])


ournames = ['Kemal', 'Syndee', 'Hobson', 'Maria']
probas = model.predict_proba(vectorizer.transform(ournames))
df_probas = pd.DataFrame(probas, index=ournames, columns=model.classes_)

training_set_names = df['name'][istrain].unique()
for n in ournames:
    print(n, n in training_set_names)

X_ournames = pd.DataFrame.sparse.from_spmatrix(
    vectorizer.transform(ournames),
    index=ournames,
    columns=vectorizer.get_feature_names_out())
X_ournames
model.predict_proba(X_ournames)
"""
>>> pd.DataFrame(model.predict_proba(X_ournames),
...     columns=model.classes_,
...     index=ournames).round(4)
...
             F       M
Kemal   0.0085  0.9915
Syndee  0.9997  0.0003
Hobson  0.0038  0.9962
Maria   1.0000  0.0000
"""

y_test = df['sex'][~istrain]
y_test_pred = model.predict(df[['len', 'year']][~istrain])
df_plot = pd.DataFrame()
df_plot['female'] = y_test
df_plot['female_pred'] = y_test_pred
y_test_proba = model.predict_proba(df[['len', 'year']][~istrain])
df_plot['female_proba'] = y_test_proba[:, 1]
df_plot.sample(30)
