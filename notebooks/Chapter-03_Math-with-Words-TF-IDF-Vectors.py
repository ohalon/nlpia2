#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-03_Math-with-Words-TF-IDF-Vectors`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc)

# #### 

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")
sentence = ('It has also arisen in criminal justice, healthcare, and '
    'hiring, compounding existing racial, economic, and gender biases.')
doc = nlp(sentence)
tokens = [token.text for token in doc]
tokens


# #### 

# In[ ]:


from collections import Counter
bag_of_words = Counter(tokens)
bag_of_words


# #### 

# In[ ]:


import pandas as pd
most_common = dict(bag_of_words.most_common())  # <1>
counts = pd.Series(most_common)  # <2>
counts


# #### 

# In[ ]:


len(counts)  # <1>


# #### 

# In[ ]:


counts.sum()


# #### 

# In[ ]:


len(tokens)  # <2>


# #### 

# In[ ]:


counts / counts.sum()  # <3>


# #### 

# In[ ]:


counts['justice']


# #### 

# In[ ]:


counts['justice'] / counts.sum()


# #### 

# In[ ]:


sentence = "Algorithmic bias has been cited in cases ranging from " \
    "election outcomes to the spread of online hate speech."
tokens = [tok.text for tok in nlp(sentence)]
counts = Counter(tokens)
dict(counts)


# #### 

# In[ ]:


from nlpia2 import wikipedia as wiki
page = wiki.page('Algorithmic Bias')  # <1>
page.content[:70]


# #### 

# In[ ]:


import requests
url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/raw/main/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response


# #### 

# In[ ]:


bias_intro_bytes = response.content  # <1>
bias_intro = response.text  # <2>
assert bias_intro_bytes.decode() == bias_intro    # <3>
bias_intro[:70]


# #### 

# In[ ]:


tokens = [tok.text for tok in nlp(bias_intro)]
counts = Counter(tokens)
counts


# #### 

# In[ ]:


counts.most_common(5)


# #### 

# In[ ]:


counts.most_common()[-4:]


# #### 

# In[ ]:


docs = [nlp(s) for s in bias_intro.split('\n')
        if s.strip()]  # <1>
counts = []
for doc in docs:
    counts.append(Counter([
        t.text.lower() for t in doc]))  # <2>
df = pd.DataFrame(counts)
df = df.fillna(0).astype(int)  # <3>
len(df)


# #### 

# In[ ]:


df.head()


# #### 

# In[ ]:


df.iloc[10]  # <1>


# #### 

# In[ ]:


docs_tokens = []
for doc in docs:
    docs_tokens.append([
        tok.text.lower() for tok in nlp(doc.text)])  # <1>
len(docs_tokens[0])


# #### 

# In[ ]:


all_doc_tokens = []
for tokens in docs_tokens:
    all_doc_tokens.extend(tokens)
len(all_doc_tokens)


# #### 

# In[ ]:


vocab  # <1>


# #### 

# In[ ]:


count_vectors = []
for tokens in docs_tokens:
    count_vectors.append(Counter(tokens))
tf = pd.DataFrame(count_vectors)  # <1>
tf = tf.T.sort_index().T
tf.fillna(0).astype(int)


# #### 

# In[ ]:


v1 = np.array(list(range(5)))
v2 = pd.Series(reversed(range(5)))
slow_answer = sum([4.2 * (x1 * x2) for x1, x2 in zip(v1, v2)])
slow_answer


# #### 

# In[ ]:


faster_answer = sum(4.2 * v1 * v2)  # <1>
faster_answer


# #### 

# In[ ]:


fastest_answer = 4.2 * v1.dot(v2)  # <2>
fastest_answer


# #### 

# In[ ]:


A.dot(B) == (np.linalg.norm(A) * np.linalg.norm(B)) * \
    np.cos(angle_between_A_and_B)


# #### 

# In[ ]:


cos_similarity_between_A_and_B = np.cos(angle_between_A_and_B) \
   = A.dot(B) / (np.linalg.norm(A) * np.linalg.norm(B))


# #### 

# In[ ]:


import math
def cosine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()] # <1>
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


# #### .Cosine similarity

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
vec1 = count_vectors[1,:]
vec2 = count_vectors[2,:]
cosine_similarity(vec1, vec2)


# #### .Cosine similarity

# In[ ]:


import copy
question = "What is algorithmic bias?"
ngram_docs = copy.copy(docs)
ngram_docs.append(question)


# #### .Cosine similarity

# In[ ]:


question_vec = vectorizer.transform([new_sentence])
question_vec


# #### .Cosine similarity

# In[ ]:


question_vec.to_array()


# #### .Cosine similarity

# In[ ]:


vocab = list(zip(*sorted((i, tok) for tok, i in
    vectorizer.vocabulary_.items())))[1]
pd.Series(question_vec.to_array()[0], index=vocab).head(8)


# #### 

# In[ ]:


cosine_similarity(count_vectors, question_vector)


# #### 

# In[ ]:


docs[3]


# #### 

# In[ ]:


ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_vectors = ngram_vectorizer.fit_transform(corpus)
ngram_vectors


# #### 

# In[ ]:


vocab = list(zip(*sorted((i, tok) for tok, i in
    ngram_vectorizer.vocabulary_.items())))[1]
pd.DataFrame(ngram_vectors.toarray(),
    columns=vocab)['algorithmic bias']


# #### 

# In[ ]:


from this import s
print(s)


# #### 

# In[ ]:


char_vectorizer = CountVectorizer(
    ngram_range=(1,1), analyzer='char')  # <1>
s_char_frequencies = char_vectorizer.fit_transform(s)
generate_histogram(
    s_char_frequencies, s_char_vectorizer)  # <2>


# #### 

# In[ ]:


DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
            '-/raw/master/src/nlpia/data')
url = DATA_DIR + '/machine_learning_full_article.txt'
ml_text = requests.get(url).content.decode()
ml_char_frequencies = char_vectorizer.fit_transform(ml_text)
generate_histogram(s_char_frequencies, s_char_vectorizer)


# #### 

# In[ ]:


chr(ord('W') - peak_distance)


# #### 

# In[ ]:


import codecs
print(codecs.decode(s, 'rot-13'))


# #### 

# In[ ]:


nltk.download('brown')  # <1>
from nltk.corpus import brown
brown.words()[:10]  # <2>


# #### 

# In[ ]:


brown.tagged_words()[:5]  # <3>


# #### 

# In[ ]:


len(brown.words())


# #### 

# In[ ]:


from collections import Counter
puncs = set((',', '.', '--', '-', '!', '?',
    ':', ';', '``', "''", '(', ')', '[', ']'))
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
token_counts.most_common(10)


# #### 

# In[ ]:


DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
            '-/raw/master/src/nlpia/data')
url = DATA_DIR + '/bias_discrimination.txt'
bias_discrimination = requests.get(url).content.decode()
intro_tokens = [token.text for token in nlp(bias_intro.lower())]
disc_tokens = [token.text for token in nlp(bias_discrimination.lower())]
intro_total = len(intro_tokens)
intro_total


# #### 

# In[ ]:


disc_total = len (disc_tokens)
disc_total


# #### 

# In[ ]:


intro_tf = {}
disc_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['bias'] = intro_counts['bias'] / intro_total
disc_counts = Counter(disc_tokens)
disc_tf['bias'] = disc_counts['bias'] / disc_total
'Term Frequency of "bias" in intro is:{:.4f}'.format(intro_tf['bias'])


# #### 

# In[ ]:


'Term Frequency of "bias" in discrimination chapter is: {:.4f}'\
    .format(disc_tf['bias'])


# #### 

# In[ ]:


intro_tf['and'] = intro_counts['and'] / intro_total
disc_tf['and'] = disc_counts['and'] / disc_total
print('Term Frequency of "and" in intro is: {:.4f}'\
    .format(intro_tf['and']))


# #### 

# In[ ]:


print('Term Frequency of "and" in discrimination chapter is: {:.4f}'\
    .format(disc_tf['and']))


# #### 

# In[ ]:


num_docs_containing_and = 0
for doc in [intro_tokens, disc_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1  # <1>


# #### 

# In[ ]:


intro_tf['black'] = intro_counts['black'] / intro_total
disc_tf['black'] = disc_counts['black'] / disc_total


# #### 

# In[ ]:


num_docs = 2
intro_idf = {}
disc_idf = {}
intro_idf['and'] = num_docs / num_docs_containing_and
disc_idf['and'] = num_docs / num_docs_containing_and
intro_idf['bias'] = num_docs / num_docs_containing_bias
disc_idf['bias'] = num_docs / num_docs_containing_bias
intro_idf['black'] = num_docs / num_docs_containing_black
disc_idf['black'] = num_docs / num_docs_containing_black


# #### 

# In[ ]:


intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['bias'] = intro_tf['bias'] * intro_idf['bias']
intro_tfidf['black'] = intro_tf['black'] * intro_idf['black']


# #### 

# In[ ]:


disc_tfidf = {}
disc_tfidf['and'] = disc_tf['and'] * disc_idf['and']
disc_tfidf['bias'] = disc_tf['bias'] * disc_idf['bias']
disc_tfidf['black'] = disc_tf['black'] * disc_idf['black']


# #### 

# In[ ]:


doc_tfidf_vectors = []
for doc in docs:  # <1>
    vec = copy.copy(zero_vector)  # <2>
    tokens = [token.text for token in nlp(doc.lower())]
    token_counts = Counter(tokens)

    for token, count in token_counts.items():
        docs_containing_key = 0
        for d in docs:
            if token in d:
                docs_containing_key += 1
        tf = value / len(vocab)
        if docs_containing_key:
            idf = len(docs) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    doc_tfidf_vectors.append(vec)


# #### 

# In[ ]:


query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)  # <1>
tokens = [token.text for token in nlp(query.lower())]
token_counts = Counter(tokens)
for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in docs:
      if key in _doc.lower():
        docs_containing_key += 1
    if docs_containing_key == 0:  # <1>
        continue
    tf = value / len(tokens)
    idf = len(docs) / docs_containing_key
    query_vec[key] = tf * idf
cosine_sim(query_vec, doc_tfidf_vectors[0])


# #### 

# In[ ]:


cosine_sim(query_vec, doc_tfidf_vectors[1])


# #### 

# In[ ]:


cosine_sim(query_vec, doc_tfidf_vectors[2])


# #### .Computing TF-IDF matrix using Scikit-Learn

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = docs
vectorizer = TfidfVectorizer(min_df=1) # <1>
vectorizer = vectorizer.fit(corpus)  # <2>
vectors = vectorizer.transform(corpus)  # <3>
print(vectors.todense().round(2))  # <4>


# #### 

# In[ ]:


DS_FAQ_URL = ('https://gitlab.com/tangibleai/qary/-/raw/main/'
    'src/qary/data/faq/faq-python-data-science-cleaned.csv')
qa_dataset = pd.read_csv(DS_FAQ_URL)


# #### 

# In[ ]:


vectorizer = TfidfVectorizer()
vectorizer.fit(df['question'])
tfidfvectors_sparse = vectorizer.transform(df['question'])  # <1>
tfidfvectors = tfidfvectors_sparse.todense()  # <2>


# #### 

# In[ ]:


def bot_reply(question):
   question_vector = vectorizer.transform([question]).todense()
   idx = question_vector.dot(tfidfvectors.T).argmax() # <1>

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


bot_reply('How do I decrease overfitting for Logistic Regression?')

