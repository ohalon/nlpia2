#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-04_Finding-Meaning-in-Word-Counts-Semantic-Analysis`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-04_Finding-Meaning-in-Word-Counts-Semantic-Analysis.adoc)

# #### .Sample weights for your topics

# In[ ]:


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


# #### 

# In[ ]:


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


# #### .The toxic comment dataset

# In[ ]:


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


# #### .The toxic comment dataset

# In[ ]:


comments.toxic.sum()


# #### .The toxic comment dataset

# In[ ]:


comments.head(6)


# #### .The toxic comment dataset

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load("en_core_web_sm")
def spacy_tokenize(sentence):
   return [token.text for token in nlp(sentence.lower())]
tfidf_model = TfidfVectorizer(tokenizer=spacy_tokenize)
tfidf_docs = tfidf_model.fit_transform(\
    raw_documents=comments.text).toarray()
tfidf_docs.shape


# #### 

# In[ ]:


mask = comments.toxic.astype(bool).values  # <1>
toxic_centroid = tfidf_docs[mask].mean(axis=0)  # <2>
nontoxic_centroid = tfidf_docs[~mask].mean(axis=0)  # <3>


# #### 

# In[ ]:


centroid_axis = toxic_centroid - nontoxic_centroid
toxicity_score = tfidf_docs.dot(centroid_axis)  # <1>
toxicity_score.round(3)


# #### 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
comments['manual_score'] = MinMaxScaler().fit_transform(\
    toxicity_score.reshape(-1,1))
comments['manual_predict'] = (comments.manual_score > .5).astype(int)
comments['toxic manual_predict manual_score'.split()].round(2).head(6)


# #### 

# In[ ]:


(1 - (comments.toxic - comments.manual_predict).abs().sum() 
    / len(comments))


# #### .LDA model performance with train-test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_docs,\
    comments.toxic.values, test_size=0.5, random_state=271828)
lda_tfidf = LDA(n_components=1)
lda = lda_tfidf.fit(X_train, y_train)  # <1>
round(float(lda.score(X_train, y_train)), 3)


# #### .LDA model performance with train-test split

# In[ ]:


round(float(lda.score(X_test, y_test)), 3)


# #### .LDA model performance with train-test split

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, lda.predict(X_test))


# #### .LDA model performance with train-test split

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lda,X_test, y_test, cmap="Greys",
               display_labels=['non-toxic', 'toxic'], colorbar=False)
plt.show()


# #### .PCA Magic

# In[ ]:


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


# #### 

# In[ ]:


tfidf_docs.shape


# #### 

# In[ ]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=16, n_iter=100)  # <1>
columns = ['topic{}'.format(i) for i in range(svd.n_components)]


# #### 

# In[ ]:


list(tfidf_model.vocabulary_.items())[:5]  # <1>


# #### 

# In[ ]:


column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),
    tfidf.vocabulary_.keys())))  # <2>
terms[:5]


# #### 

# In[ ]:


topic_term_matrix = pd.DataFrame(
    svd.components_, columns=terms,
    index=['topic{}'.format(i) for i in range(16)])
pd.options.display.max_columns = 8
topic_term_matrix.sample(5, axis='columns',
    random_state=271828).head(4)  # <1>


# #### 

# In[ ]:


pd.options.display.max_columns = 8
toxic_terms = topic_term_matrix[
    'pathetic crazy stupid idiot lazy hate die kill'.split()
    ].round(3) * 100  # <1>
toxic_terms


# #### 

# In[ ]:


toxic_terms.T.sum()


# #### 

# In[ ]:


tfidf_docs = tfidf_docs - tfidf_docs.mean()


# #### 

# In[ ]:


X_train_16d, X_test_16d, y_train_16d, y_test_16d = train_test_split(
    svd_topic_vectors, comments.toxic.values, test_size=0.5,
    random_state=271828)
lda_lsa = LinearDiscriminantAnalysis(n_components=1)
lda_lsa = lda_lsa.fit(X_train_16d, y_train_16d)
round(float(lda_lsa.score(X_train_16d, y_train_16d)), 3)


# #### 

# In[ ]:


round(float(lda_lsa.score(X_test_16d, y_test_16d)), 3)


# #### 

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test_16d, lda_lsa.predict(X_test_16d).round(3)


# #### 

# In[ ]:


hparam_table = pd.DataFrame()
tfidf_performance = {'classifier': 'LDA',
                     'features': 'tf-idf (spacy tokenizer)',
                     'train_accuracy': 0.99 ,
                     'test_accuracy': 0.554,
                     'test_precision': 0.383 ,


# #### .A function that creates a record in hyperparameter table.

# In[ ]:


def hparam_rec(model, X_train, y_train, X_test, y_test,
               model_name, features):
    return {
        'classifier': model_name,
        'features': features,
        'train_accuracy': float(model.score(X_train, y_train)),
        'test_accuracy': float(model.score(X_test, y_test)),
        'test_precision':
            precision_score(y_test, model.predict(X_test)),
        'test_recall':
            recall_score(y_test, model.predict(X_test)),
        'test_f1': f1_score(y_test, model.predict(X_test))
        }
lsa_performance = hparam_rec(lda_lsa, X_train_16d, y_train_16d,
       X_test_16d,y_test_16d, 'LDA', 'LSA (16 components)'))
hparam_table = hparam_table.append(lsa_performance)
hparam_table.T  # <1>


# #### .A function that creates a record in hyperparameter table.

# In[ ]:


def evaluate_model(X,y, classifier, classifier_name, features):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=271828)
    classifier = classifier.fit(X_train, y_train)
    return hparam_rec(classifier, X_train, y_train, X_test,y_test,
                      classifier_name, features)


# #### 

# In[ ]:


total_corpus_len = 0
for document_text in comments.text:
    total_corpus_len += len(spacy_tokenize(document_text))
mean_document_len = total_corpus_len / len(sms)
round(mean_document_len, 2)


# #### 

# In[ ]:


sum([len(spacy_tokenize(t)) for t in comments.text]
    ) * 1. / len(comments.text)


# #### 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer(tokenizer=spacy_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(


# #### 

# In[ ]:


column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),
    counter.vocabulary_.keys())))


# #### 

# In[ ]:


comments.loc['comment0'].text


# #### 

# In[ ]:


bow_docs.loc['comment0'][bow_docs.loc['comment0'] > 0].head()


# #### 

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)  # <1>
ldia.components_.shape


# #### 

# In[ ]:


pd.set_option('display.width', 75)
term_topic_matrix = pd.DataFrame(ldia.components_, index=terms,\ 
    columns=columns)  # <1>
term_topic_matrix.round(2).head(3)


# #### 

# In[ ]:


toxic_terms= components.loc['pathetic crazy stupid lazy idiot hate die kill'.split()].round(2)
toxic_terms


# #### 

# In[ ]:


non_trivial_terms = [term for term in components.index


# #### 

# In[ ]:


ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors,\
    index=index, columns=columns)
ldia16_topic_vectors.round(2).head()


# #### 

# In[ ]:


model_ldia16 = LinearDiscriminantAnalysis()
ldia16_performance=evaluate_model(ldia16_topic_vectors,


# #### 

# In[ ]:


hparam_table = hparam_table.append(ldia16_performance,
   ignore_index = True)
hparam_table.T


# #### 

# In[ ]:


ldia32 = LDiA(n_components=32, learning_method='batch')
ldia32 = ldia32.fit(bow_docs)
model_ldia32 = LinearDiscriminantAnalysis()
ldia32_performance =evaluate_model(ldia32_topic_vectors,
         comments.toxic, model_ldia32, 'LDA', 'LDIA (32d)')
hparam_table = hparam_table.append(ldia32_performance,
          ignore_index = True)
hparam_table.T


# #### 

# In[ ]:


import sklearn
sklearn.__file__


# #### 

# In[ ]:


from sklearn.discriminant_analysis\
    import LinearDiscriminantAnalysis as LDA
get_ipython().run_line_magic('pinfo2', 'LDA')


# #### 

# In[ ]:


similarity = 1. / (1. + distance)
distance = (1. / similarity) - 1.


# #### 

# In[ ]:


similarity = 1. - distance
distance = 1. - similarity


# #### 

# In[ ]:


import math
angular_distance = math.acos(cosine_similarity) / math.pi
distance = 1. / similarity - 1.
similarity = 1. - distance


# #### 

# In[ ]:


REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/master'
FAQ_DIR = 'src/qary/data/faq'
FAQ_FILENAME = 'short-faqs.csv'
DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])
df = pd.read_csv(DS_FAQ_URL)


# #### 

# In[ ]:


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


# #### 

# In[ ]:


bot_reply("What's overfitting a model?")


# #### 

# In[ ]:


bot_reply("How do I decrease overfitting for Logistic Regression?")

