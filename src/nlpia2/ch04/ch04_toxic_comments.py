import pandas as pd
pd.options.display.width = 120  # <1>
pd.set_option('display.max_columns', 7)

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/-/raw/master/src/nlpia/data')
url= DATA_DIR + '/toxic_comment_small.csv'
comments = pd.read_csv(url)
index = ['comment{}{}'.format(i, '!'*j) for (i,j) in zip(range(len(comments)), comments.toxic)]  # <2>
comments = pd.DataFrame(comments.values, columns=comments.columns, index=index)
mask = comments.toxic.astype(bool).values
comments['toxic'] = comments.toxic.astype(int)
"""
>>> comments.head(6)
                                                        text  toxic
comment0   you have yet to identify where my edits violat...      0
comment1   "\n as i have already said,wp:rfc or wp:ani. (...      0
comment2   your vote on wikiquote simple english when it ...      0
comment3   your stalking of my edits i've opened a thread...      0
comment4!  straight from the smear site itself. the perso...      1
comment5   no, i can't see it either - and i've gone back...      0
"""
#Preprocessing




from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(sentence):
    return [token.text for token in nlp(sentence.lower())]

tfidf = TfidfVectorizer(tokenizer=spacy_tokenizer)
tfidf_docs = tfidf.fit_transform(raw_documents=comments.text).toarray()
"""
>>> tfidf_docs.shape
(5000, 25172)
>>> comments.toxic.sum()
650
"""

mask = comments.toxic.astype(bool).values  # <1>
toxic_centroid = tfidf_docs[mask].mean(axis=0) # <2>
nontoxic_centroid = tfidf_docs[~mask].mean(axis=0)

toxicity_score = tfidf_docs.dot(toxic_centroid - nontoxic_centroid)
"""
>>> toxicity_score
array([-0.01469806, -0.02007376,  0.03856095, ..., -0.01014774, -0.00344281,  0.00395752])
"""

from sklearn.preprocessing import MinMaxScaler
comments['manual_score'] = MinMaxScaler().fit_transform(toxicity_score.reshape(-1, 1))
comments['manual_predict'] = (comments.manual_score > .5).astype(int)
"""
>>> comments['toxic manual_predict manual_score'.split()].round(2).head(6)
           toxic  manual_predict  manual_score
comment0       0               0          0.41
comment1       0               0          0.27
comment2       0               0          0.35
comment3       0               0          0.47
comment4!      1               0          0.48
comment5       0               0          0.31

"""

(1. - (comments.toxic - comments.manual_predict).abs().sum() / len(comments)).round(3)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_tfidf = LinearDiscriminantAnalysis()
lda_tfidf = lda_tfidf.fit(tfidf_docs, comments['toxic'])  # <1>
comments['tfidf_predict'] = lda_tfidf.predict(tfidf_docs)
"""
round(float(lda_tfidf.score(tfidf_docs, comments['toxic'])), 3)
0.999
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, \
                comments.toxic.values, test_size=0.5, random_state=271828) # <1>
lda_tfidf_train = LinearDiscriminantAnalysis(n_components=1)
lda_tfidf_train = lda_tfidf_train.fit(X_train, y_train)  # <2>
"""
>>> round(float(lda_tfidf_train.score(X_train, y_train)), 3)
0.99
>>> round(float(lda_tfidf_train.score(X_test, y_test)), 3)
0.554
"""

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, lda_tfidf_train.predict(X_test))
"""
array([[1261,  913],
       [ 201,  125]], dtype=int64)
"""

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lda_tfidf_train,X_test, y_test, cmap="Greys",
                      display_labels=['non-toxic', 'toxic'], colorbar=False)
#plt.show()

from sklearn.metrics import f1_score
f1_score(y_test,lda_tfidf_train.predict(X_test))
"""
0.1832844574780059
"""


### Latent Semantic Analysis
from sklearn.decomposition import TruncatedSVD
tfidf_docs_centered = tfidf_docs - tfidf_docs.mean()
svd = TruncatedSVD(n_components=16, n_iter=100)  # <1>
columns = ['topic{}'.format(i) for i in range(svd.n_components)]
svd_topic_vectors = svd.fit_transform(tfidf_docs)
svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns,
                  index=index)
"""
>>> svd_topic_vectors.round(3).head(6)
           topic0  topic1  topic2  topic3  ...  topic12  topic13  topic14  topic15
comment0    0.121  -0.055   0.036  -0.040  ...    0.013   -0.038    0.089    0.011
comment1    0.215   0.141  -0.006  -0.006  ...   -0.040    0.079   -0.016   -0.070
comment2    0.342  -0.200   0.044  -0.070  ...    0.059   -0.138    0.023    0.069
comment3    0.130  -0.074   0.034  -0.018  ...    0.119   -0.060    0.014    0.073
comment4!   0.166  -0.081   0.040   0.136  ...    0.066   -0.008    0.063   -0.020
comment5    0.256  -0.122  -0.055   0.082  ...    0.011    0.093   -0.083   -0.074

"""

"""
>>> list(tfidf_model.vocabulary_.items())[:5] #<1>
[('you', 18890),
 ('have', 8093),
 ('yet', 18868),
 ('to', 17083),
 ('identify', 8721)]
"""

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),
     tfidf.vocabulary_.keys())))  # <2>
"""
>>> terms
('\n', '\n ', '\n \n', '\n \n ', '\n  ')
"""
topic_term_matrix = pd.DataFrame(svd.components_, columns=terms,
                   index=['topic{}'.format(i) for i in range(16)])
"""
>>> pd.options.display.max_columns = 8
>>> topic_term_matrix.head(4).round(3)
>>> toxic_terms= topic_term_matrix['pathetic crazy stupid lazy idiot hate die kill'.split()].round(3) * 100
>>> 

"""

X_train_16d, X_test_16d, y_train_16d, y_test_16d = train_test_split(svd_topic_vectors, \
                                                    comments.toxic.values, test_size=0.5, random_state=271828)
lda_lsa = LinearDiscriminantAnalysis(n_components=1)
lda_lsa = lda_lsa.fit(X_train_16d, y_train_16d)  # <2>
"""
>>> round(float(lda_lsa.score(X_train_16d, y_train_16d)), 3)
0.881
>>> round(float(lda_lsa.score(X_test_16d, y_test_16d)), 3)
0.88
"""
from sklearn.metrics import f1_score
f1_score(y_test_16d, lda_lsa.predict(X_test_16d))
"""
# comparing to PCA 
from sklearn.decomposition import PCA
pca_model = PCA(n_components=16)
tfidf_docs_16d = pca_model.fit_transform(tfidf_docs)


from sklearn.model_selection import train_test_split
X_train_16d, X_test_16d, y_train_16d, y_test_16d = train_test_split(tfidf_docs_16d, \
                                                    comments.toxic.values, test_size=0.5, random_state=271828)
lda_lsa = LinearDiscriminantAnalysis(n_components=1)
lda_lsa = lda_lsa.fit(X_train_16d, y_train_16d)  # <2>
round(float(lda_lsa.score(X_train_16d, y_train_16d)), 3)
round(float(lda_lsa.score(X_test_16d, y_test_16d)), 3)
"""

# Hyperparameter table
hparam_table = pd.DataFrame()

tfidf_performance = {'classifier': 'LDA',
                     'features': 'tf-idf (spacy tokenizer)',
                     'train_accuracy': 0.99 ,
                     'test_accuracy': 0.554,
                     'test_precision': 0.383 ,
                     'test_recall': 0.12,
                     'test_f1': 0.183}

hparam_table = hparam_table.append(tfidf_performance, ignore_index=True)

from sklearn.metrics import precision_score, recall_score, f1_score

def hparam_rec(model, X_train, y_train, X_test, y_test, classifier_name, features):
    return {'classifier': classifier_name,
            'features': features,
            'train_accuracy': float(model.score(X_train, y_train)),
            'test_accuracy': float(model.score(X_test, y_test)),
            'test_precision': precision_score(y_test, model.predict(X_test)),
            'test_recall': recall_score(y_test, model.predict(X_test)),
            'test_f1': f1_score(y_test, model.predict(X_test)) }

lsa_performance = hparam_rec(lda_lsa, X_train_16d, y_train_16d, X_test_16d,y_test_16d, 'LDA', 'LSA (16d)')
hparam_table = hparam_table.append(lsa_performance,ignore_index=True)

def evaluate_model(X,y, classifier, classifier_name, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=271828)
    classifier = classifier.fit(X_train, y_train)
    return hparam_rec(classifier, X_train, y_train, X_test,y_test,
                      classifier_name, features)

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer(tokenizer=spacy_tokenizer)
counter = counter.fit(comments.text)

bow_docs = pd.DataFrame(counter.transform(comments.text)
                        .toarray(), index=index)
column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
                                     counter.vocabulary_.keys())))
bow_docs.columns = terms

"""
>>> comments.loc['comment0'].text
"""


from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)  # <1>
ldia.components_.shape
columns = ['topic{}'.format(i) for i in range(16)]
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,
                                    index=index, columns=columns)

components = pd.DataFrame(ldia.components_.T, index=terms,
    columns=columns)
components.round(2).head(3)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model_ldia16 = LinearDiscriminantAnalysis()
ldia16_performance =evaluate_model(ldia16_topic_vectors, comments.toxic, model_ldia16, 'LDA', 'LDIA (16d)')

hparam_table = hparam_table.append(ldia16_performance, ignore_index = True)

ldia32 = LDiA(n_components=32, learning_method='batch')
ldia32_topic_vectors = ldia.fit_transform(bow_docs)
model_ldia32 = LinearDiscriminantAnalysis()
ldia32_performance =evaluate_model(ldia32_topic_vectors, comments.toxic, model_ldia32, 'LDA', 'LDIA (32d)')

hparam_table = hparam_table.append(ldia32_performance, ignore_index = True)

words = counter.get_feature_names()

REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/master'
FAQ_DIR = 'src/qary/data/faq'
FAQ_FILENAME = 'short-faqs.csv'
DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])

df = pd.read_csv(DS_FAQ_URL)

vectorizer = TfidfVectorizer()
df['questions+answers'] = df['question'] + df['answer']
vectorizer.fit(df['questions+answers'])
# vectorize all the questions/answers in qa_dataset
tfidfvectors_sparse = vectorizer.transform(df['question'])
tfidfvectors = tfidfvectors_sparse.todense()
svd = TruncatedSVD(n_components=16, n_iter=100)
tfidfvectors_16d = svd.fit_transform(tfidfvectors)

def bot_reply(question):
      question_tfidf = vectorizer.transform([question]).todense()
      question_16d = svd.transform(question_tfidf)
      idx = question_16d.dot(tfidfvectors_16d.T).argmax()
      print(
            f"Your question:\n  {question}\n\n"
            f"Most similar FAQ question:\n  {df['question'][idx]}\n\n"
            f"Answer to that FAQ question:\n  {df['answer'][idx]}\n\n"
           )

bot_reply("How do I decrease overfitting for Logistic Regression?")


for topic_idx, topic in enumerate(ldia.components_):
          print(f"\nTopic #{topic_idx + 1}:")
          print("; ".join([words[i]
                           for i in topic.argsort()[:-6:-1]]))