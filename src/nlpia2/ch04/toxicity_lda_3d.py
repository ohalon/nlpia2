import pandas as pd
pd.options.display.width = 120  # <1>

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/-/raw/master/src/nlpia/data')
url= DATA_DIR + '/sms-spam.csv'
sms = pd.read_csv(url)
index = ['sms{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]  # <2>
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
mask = sms.spam.astype(bool).values
sms['spam'] = sms.spam.astype(int)
"""
>>> sms.head(6)
       spam                                               text
sms0      0  Go until jurong point, crazy.. Available only ...
sms1      0                      Ok lar... Joking wif u oni...
sms2!     1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3      0  U dun say so early hor... U c already then say...
sms4      0  Nah I don't think he goes to usf, he lives aro...
sms5!     1  FreeMsg Hey there darling it's been 3 week's n...
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    return [token.text for token in nlp(sentence.lower())]

tfidf_model = TfidfVectorizer(tokenizer=spacy_tokenizer)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
"""

mask = sms.spam.astype(bool).values  # <1>
spam_centroid = tfidf_docs[mask].mean(axis=0) # <2>
ham_centroid = tfidf_docs[~mask].mean(axis=0)
"""
>>> mask = sms.spam.astype(bool)
>>> spam_centroid = tfidf_docs[mask].mean(axis=0)
>>> spam_centroid.round(2)
array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
>>> ham_centroid = tfidf_docs[~mask].mean(axis=0)
>>> ham_centroid.round(2)
array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])
"""

spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
"""
>>> spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
>>> spamminess_score
array([-0.01469806, -0.02007376,  0.03856095, ..., -0.01014774, -0.00344281,  0.00395752])
"""

from sklearn.preprocessing import MinMaxScaler
sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
sms['lda_predict'] = (sms.lda_score > .5).astype(int)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_tfidf = LinearDiscriminantAnalysis()
lda_tfidf = lda_tfidf.fit(tfidf_docs, sms['spam'])  # <1>
sms['lda_tfidf_predict'] = lda_tfidf.predict(tfidf_docs)
round(float(lda_tfidf.score(tfidf_docs, sms['spam'])), 3)

"""
>>> from sklearn.preprocessing import MinMaxScaler
>>> sms['lda_score'] = MinMaxScaler().fit_transform(spamminess_score.reshape(-1,1))
>>> sms['lda_predict'] = (sms.lda_score > .5).astype(int)
>>> sms['spam lda_predict lda_score'.split()].round(2).head(6)
       spam  lda_predict  lda_score
sms0      0            0       0.23
sms1      0            0       0.18
sms2!     1            1       0.72
sms3      0            0       0.18
sms4      0            0       0.29
sms5!     1            1       0.55
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_docs,\
     sms.spam.values, test_size=0.5, random_state=271828)
lda_tfidf = lda_tfidf.fit(X_train, y_train)  # <2>
round(float(lda_tfidf.score(X_train, y_train)), 3)
round(float(lda_tfidf.score(X_test, y_test)), 3)

# running Hobson's code from previous book

from nltk.tokenize.casual import casual_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    return [token.text for token in nlp(sentence.lower())]

tfidf = TfidfVectorizer(tokenizer=spacy_tokenizer)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = tfidf_docs - tfidf_docs.mean(axis=0)

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, sms.spam.values, test_size = 0.5, random_state = 271828)
lda = LinearDiscriminantAnalysis(n_components=1)
lda = lda.fit(X_train, y_train)



from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn
pca_model = PCA(n_components=3)
tfidf_docs_3d = pca_model.fit_transform(tfidf_docs)
df = pd.DataFrame(tfidf_docs_3d)
ax = df[~mask].plot(x=0, y=1, kind='scatter', alpha=.5, c='green')
df[mask].plot(x=0, y=1, ax=ax, alpha=.1, kind='scatter', c='red')
plt.xlabel(' x')
plt.ylabel(' y')
plt.show()

"""
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn
pca_model = PCA(n_components=3)
tfidf_docs_3d = pca_model.fit_transform(tfidf_docs)
df = pd.DataFrame(tfidf_docs_3d)

import plotly as py
spam_trace = dict(
        x=df[0][mask], y=df[1][mask], z=df[2][mask],
        type="scatter3d", mode='markers',
        marker= dict(size=3, color='red', line=dict(width=0))
    )
ham_trace = dict(
        x=df[0][~mask], y=df[1][~mask], z=df[2][~mask],
        type="scatter3d", mode='markers',
        marker= dict(size=3, color='green', line=dict(width=0))
    )
fig = dict(data=[ham_trace, spam_trace], layout={'title': 'LDA Spamminess Model'})
py.offline.plot(fig, filename='lda_toxicity_3d_scatter.html')
"""