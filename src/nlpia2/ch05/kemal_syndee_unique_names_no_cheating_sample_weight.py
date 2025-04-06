import numpy as np
import pandas as pd
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
hist -o -p
history
hist -o -p
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(names.index.values[istrain])
vecs = vectorizer.transform(names.index)
from sklearn.feature_extraction.text TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(names.index.values[istrain])
vecs = vectorizer.transform(names.index)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(df['name'][istrain])
vectorizer
vecs = vectorizer.transform(df['names'])
vecs = vectorizer.transform(df['name'])
vecs[istrain]
dfvecs = pd.DataFrame(vecs, dtype=pd.SparseDtype("float"))
pd.SparseDataFrame
dfvecs = pd.DataFrame.sparse.from_spmatrix(vecs)
dfvecs.columns = vectorizer.get_feature_names_out()
dfvecs
dfvecs.head(2).T.head(100)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
vectorizer.fit(df['name'][istrain])
vectorizer
vecs = vectorizer.transform(df['name'])
dfvecs = pd.DataFrame.sparse.from_spmatrix(vecs)
dfvecs.head(2).T.head(100)
dfvecs.columns = vectorizer.get_feature_names_out()
dfvecs.head(5).T.head(100)
df['name'].head()
dfvecs.index = names
dfvecs.index = df['name']
dfvecs.head(5).T.head(100)
dfvecs['Amy'].head(5).T.head(100)
dfvecs[['Amy']].T.head(100)
dfvecs[['Amy']]
dfvecs
dfvecs.head()
dfvecs.head().T
dfvecs.T[['Amy', 'Alex']]
dfvecs.T[['Amy', 'Alex']].head().T
hist
X = dfvecs
model = LogisticRegression()
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model = LogisticRegression(max_iterations=5000)
model = LogisticRegression(C=.5, max_iterations=5000)
model = LogisticRegression(C=.5, max_iter=5000)
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[~istrain], y[~istrain], sample_weight=df['count'][~istrain])
model.score(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model = LogisticRegression(C=10, max_iter=5000)
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[~istrain], y[~istrain], sample_weight=df['count'][~istrain])
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.fit(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[istrain], y[istrain], sample_weight=df['count'][istrain])
model.score(X[~istrain], y[~istrain], sample_weight=df['count'][~istrain])
hist -o -p
model.classes_
model.predict_proba(vectorizer.transform(['Kemal', 'Syndee', 'Hobson', 'Maria']))
pd.DataFrame(model.predict_proba(vectorizer.transform(['Kemal', 'Syndee', 'Hobson', 'Maria'])), columns = model.classes_)
pd.DataFrame(model.predict_proba(vectorizer.transform(['Kemal', 'Syndee', 'Hobson', 'Maria'])),columns = model.classes_)
pd.DataFrame(model.predict_proba(vectorizer.transform(['Kemal', 'Syndee', 'Hobson', 'Maria'])),
columns = model.classes_, index=['Kemal', 'Syndee', 'Hobson', 'Maria'])
'Kemal' in names
'Syndee' in names
'Hobson' in names
'Maria' in names
'Maria' in names[istrain]
'Maria' in df['name'][istrain].unique()
hist
df['name'][istrain].unique()
ournames = ['Kemal', 'Syndee', 'Hobson', 'Maria']
probas = model.predict_proba(vectorizer.transform(ournames))
df_probas = pd.DataFrame(probas, index=ournames, columns=model.classes_)

training_set_names = df['name'][istrain].unique()
for n in ournames:
    print(n, n in training_set_names)
vectorizer.transform(ournames)
model.predict_proba(vectorizer.transform(ournames))
model.predict_proba(vectorizer.transform(ournames))
model.predict_proba(vectorizer.transform(ournames))
model.predict_proba(vectorizer.transform(ournames), feature_names=vectorizer.get_feature_names_out())
model.feature_names_in_
X_ournames = vectorizer.transform(ournames)
X.shape
X_ournames.shape
type(X)
type(X_ournames)
X_ournames = pd.DataFrame.sparse.from_spmatrix(vectorizer.transform(ournames), index=ournames, columns=vectorizer.get_feature_names_in())
X_ournames = pd.DataFrame.sparse.from_spmatrix(vectorizer.transform(ournames), index=ournames, columns=vectorizer.get_feature_names_out())
X_ournames
model.predict_proba(X_ournames)
hist
hist -o -p
pd.DataFrame(model.predict_proba(X_ournames), columns=model.classes_, index=ournames)
hist -o -p
pd.DataFrame(model.predict_proba(X_ournames), 
    columns=model.classes_,
    index=ournames).round(3)
pd.DataFrame(model.predict_proba(X_ournames), 
    columns=model.classes_,
    index=ournames).round(4)
hist -o -p
hist -o -p -f kemal_syndee_unique_names_no_cheating_sample_weight.md
hist -f kemal_syndee_unique_names_no_cheating_sample_weight.py
