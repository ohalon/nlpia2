import numpy as np

topic = {}

tfidf = dict(list(zip('cat dog apple lion NYC love'.split(),
    np.random.rand(6))))  # <1>

topic['petness'] = (.3 * tfidf['cat'] +\
                    .3 * tfidf['dog'] +\
                     0 * tfidf['apple'] +\
                     0 * tfidf['lion'] -\
                    .2 * tfidf['NYC'] +\
                    .2 * tfidf['love'])  # <2>

topic['animalness']  = (.1 * tfidf['cat']  +\
                        .1 * tfidf['dog'] -\
                        .1 * tfidf['apple'] +\
                        .5 * tfidf['lion'] +\
                        .1 * tfidf['NYC'] -\
                        .1 * tfidf['love'])

topic['cityness']    = ( 0 * tfidf['cat']  -\
                        .1 * tfidf['dog'] +\
                        .2 * tfidf['apple'] -\
                        .1 * tfidf['lion'] +\
                        .5 * tfidf['NYC'] +\
                        .1 * tfidf['love'])

word_vector = {}

word_vector['cat']  =  .3*topic['petness'] +\
                       .1*topic['animalness'] +\
                        0*topic['cityness']

word_vector['dog']  =  .3*topic['petness'] +\
                       .1*topic['animalness'] -\
                       .1*topic['cityness']

word_vector['apple']=   0*topic['petness'] -\
                       .1*topic['animalness'] +\
                       .2*topic['cityness']

word_vector['lion'] =   0*topic['petness'] +\
                       .5*topic['animalness'] -\
                       .1*topic['cityness']

word_vector['NYC']  = -.2*topic['petness'] +\
                       .1*topic['animalness'] +\
                       .5*topic['cityness']

word_vector['love'] =  .2*topic['petness'] -\
                       .1*topic['animalness'] +\
                       .1*topic['cityness']

import pandas as pd

pd.options.display.width = 120  # <1>

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/-/raw/master/'
            'src/nlpia/data')

url= DATA_DIR + '/toxic_comment_small.csv'

comments = pd.read_csv(url)

index = ['comment{}{}'.format(i, '!'*j) for (i,j) in
         zip(range(len(comments)), comments.toxic)
        ]  # <2>

comments = pd.DataFrame(
    comments.values, columns=comments.columns, index=index)

mask = comments.toxic.astype(bool).values

comments['toxic'] = comments.toxic.astype(int)

len(comments)

comments.toxic.sum()

comments.head(6)

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_tokenize(sentence):
   return [token.text for token in nlp(sentence.lower())]

tfidf_model = TfidfVectorizer(tokenizer=spacy_tokenize)

tfidf_docs = tfidf_model.fit_transform(\
    raw_documents=comments.text).toarray()

tfidf_docs.shape

mask = comments.toxic.astype(bool).values  # <1>

toxic_centroid = tfidf_docs[mask].mean(axis=0)  # <2>

nontoxic_centroid = tfidf_docs[~mask].mean(axis=0)  # <3>

centroid_axis = toxic_centroid - nontoxic_centroid

toxicity_score = tfidf_docs.dot(centroid_axis)  # <1>

toxicity_score.round(3)

from sklearn.preprocessing import MinMaxScaler

comments['manual_score'] = MinMaxScaler().fit_transform(\
    toxicity_score.reshape(-1,1))

comments['manual_predict'] = (comments.manual_score > .5).astype(int)

comments['toxic manual_predict manual_score'.split()].round(2).head(6)

(1 - (comments.toxic - comments.manual_predict).abs().sum() 
    / len(comments))

from sklearn import discriminant_analysis

lda_tfidf = discriminant_analysis.LinearDiscriminantAnalysis

lda_tfidf = lda_tfidf.fit(tfidf_docs, comments['toxic'])

comments['tfidf_predict'] = lda_tfidf.predict(tfidf_docs)

float(lda_tfidf.score(tfidf_docs, comments['toxic']))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tfidf_docs,\
    comments.toxic.values, test_size=0.5, random_state=271828)

lda_tfidf = LDA(n_components=1)

lda = lda_tfidf.fit(X_train, y_train)  # <1>

round(float(lda.score(X_train, y_train)), 3)

round(float(lda.score(X_test, y_test)), 3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, lda.predict(X_test))

import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(lda,X_test, y_test, cmap="Greys",
               display_labels=['non-toxic', 'toxic'], colorbar=False)

plt.show()

import pandas as pd

pd.set_option('display.max_columns', 6)  # <1>

from sklearn.decomposition import PCA

import seaborn

from matplotlib import pyplot as plt

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
            '-/raw/master/src/nlpia/data')

df = pd.read_csv(DATA_DIR + '/pointcloud.csv.gz', index_col=0)

pca = PCA(n_components=2)  # <3>

df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))

df2d.plot(kind='scatter', x='x', y='y')

plt.show()

tfidf_docs.shape

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=16, n_iter=100)  # <1>

columns = ['topic{}'.format(i) for i in range(svd.n_components)]

svd_topic_vectors = svd.fit_transform(tfidf_docs)  # <2>

svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns,\
    index=index)

svd_topic_vectors.round(3).head(6)

list(tfidf_model.vocabulary_.items())[:5]  # <1>

column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),
    tfidf.vocabulary_.keys())))  # <2>

terms[:5]

topic_term_matrix = pd.DataFrame(
    svd.components_, columns=terms,
    index=['topic{}'.format(i) for i in range(16)])

pd.options.display.max_columns = 8

topic_term_matrix.sample(5, axis='columns',
    random_state=271828).head(4)  # <1>

pd.options.display.max_columns = 8

toxic_terms = topic_term_matrix[
    'pathetic crazy stupid idiot lazy hate die kill'.split()
    ].round(3) * 100  # <1>

toxic_terms

toxic_terms.T.sum()

tfidf_docs = tfidf_docs - tfidf_docs.mean()

X_train_16d, X_test_16d, y_train_16d, y_test_16d = train_test_split(
    svd_topic_vectors, comments.toxic.values, test_size=0.5,
    random_state=271828)

lda_lsa = LinearDiscriminantAnalysis(n_components=1)

lda_lsa = lda_lsa.fit(X_train_16d, y_train_16d)

round(float(lda_lsa.score(X_train_16d, y_train_16d)), 3)

round(float(lda_lsa.score(X_test_16d, y_test_16d)), 3)

from sklearn.metrics import f1_score

f1_score(y_test_16d, lda_lsa.predict(X_test_16d).round(3)

hparam_table = pd.DataFrame()

tfidf_performance = {'classifier': 'LDA',
                     'features': 'tf-idf (spacy tokenizer)',
                     'train_accuracy': 0.99 ,
                     'test_accuracy': 0.554,
                     'test_precision': 0.383 ,
                     'test_recall': 0.12,
                     'test_f1': 0.183}

hparam_table = hparam_table.append(
    tfidf_performance, ignore_index=True)  # <1>

def hparam_rec(model, X_train, y_train, X_test, y_test,

lsa_performance = hparam_rec(lda_lsa, X_train_16d, y_train_16d,
       X_test_16d,y_test_16d, 'LDA', 'LSA (16 components)'))

hparam_table = hparam_table.append(lsa_performance)

hparam_table.T  # <1>

def evaluate_model(X,y, classifier, classifier_name, features):
 X_train, X_test, y_train, y_test = train_test_split(X, y,

total_corpus_len = 0

for document_text in comments.text:
    total_corpus_len += len(spacy_tokenize(document_text))

mean_document_len = total_corpus_len / len(sms)

round(mean_document_len, 2)

sum([len(spacy_tokenize(t)) for t in comments.text]) * 1. /

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer(tokenizer=spacy_tokenize)

bow_docs = pd.DataFrame(counter.fit_transform(

column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
    counter.vocabulary_.keys())))

bow_docs.columns = terms

comments.loc['comment0'].text

bow_docs.loc['comment0'][bow_docs.loc['comment0'] > 0].head()

from sklearn.decomposition import LatentDirichletAllocation as LDiA

ldia = LDiA(n_components=16, learning_method='batch')

ldia = ldia.fit(bow_docs)  # <1>

ldia.components_.shape

pd.set_option('display.width', 75)

term_topic_matrix = pd.DataFrame(ldia.components_, index=terms,\ 
    columns=columns)  # <1>

term_topic_matrix.round(2).head(3)

toxic_terms= components.loc['pathetic crazy stupid lazy idiot hate die kill'.split()].round(2)

toxic_terms

non_trivial_terms = [term for term in components.index

ldia16_topic_vectors = ldia.transform(bow_docs)

ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,\
    index=index, columns=columns)

ldia16_topic_vectors.round(2).head()

model_ldia16 = LinearDiscriminantAnalysis()

ldia16_performance=evaluate_model(ldia16_topic_vectors,

hparam_table = hparam_table.append(ldia16_performance,
   ignore_index = True)

hparam_table.T

ldia32 = LDiA(n_components=32, learning_method='batch')

ldia32 = ldia32.fit(bow_docs)

model_ldia32 = LinearDiscriminantAnalysis()

ldia32_performance =evaluate_model(ldia32_topic_vectors,
         comments.toxic, model_ldia32, 'LDA', 'LDIA (32d)')

hparam_table = hparam_table.append(ldia32_performance,
          ignore_index = True)

hparam_table.T

import sklearn

sklearn.__file__

from sklearn.discriminant_analysis\
    import LinearDiscriminantAnalysis as LDA

LDA??

similarity = 1. / (1. + distance)

distance = (1. / similarity) - 1.

similarity = 1. - distance

distance = 1. - similarity

import math

angular_distance = math.acos(cosine_similarity) / math.pi

distance = 1. / similarity - 1.

similarity = 1. - distance

REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/master'

FAQ_DIR = 'src/qary/data/faq'

FAQ_FILENAME = 'short-faqs.csv'

DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])

df = pd.read_csv(DS_FAQ_URL)

vectorizer = TfidfVectorizer()

vectorizer.fit(df['question'])

tfidfvectors = vectorizer.transform(df['question'])

svd = TruncatedSVD(n_components=16, n_iterations=100)

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

bot_reply("What's overfitting a model?")

bot_reply("How do I decrease overfitting for Logistic Regression?")
