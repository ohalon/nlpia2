#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-03_Math-with-Words-TF-IDF-Vectors`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc)

# #### 

# In[1]:


import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
sentence = ('It has also arisen in criminal justice, healthcare, and '
    'hiring, compounding existing racial, economic, and gender biases.')
doc = nlp(sentence)
tokens = [token.text for token in doc]
tokens


# #### 

# In[2]:


from collections import Counter
bag_of_words = Counter(tokens)
bag_of_words


# #### 

# In[3]:


bag_of_words.most_common(3)  # <1>


# #### 

# In[4]:


import pandas as pd
most_common = dict(bag_of_words.most_common())  # <1>
counts = pd.Series(most_common)  # <2>
counts


# #### 

# In[5]:


len(counts)  # <1>


# #### 

# In[6]:


counts.sum()


# #### 

# In[7]:


len(tokens)  # <2>


# #### 

# In[8]:


counts / counts.sum()  # <3>


# #### 

# In[9]:


counts['justice']


# #### 

# In[10]:


counts['justice'] / counts.sum()


# #### 

# In[11]:


sentence = "Algorithmic bias has been cited in cases ranging from " \
    "election outcomes to the spread of online hate speech."
tokens = [tok.text for tok in nlp(sentence)]
counts = Counter(tokens)
dict(counts)


# #### 

# In[12]:


import requests
url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/raw/main/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response


# #### 

# In[13]:


bias_intro_bytes = response.content  # <1>
bias_intro = response.text  # <2>
assert bias_intro_bytes.decode() == bias_intro    # <3>
bias_intro[:70]


# #### 

# In[14]:


tokens = [tok.text for tok in nlp(bias_intro)]
counts = Counter(tokens)
counts


# #### 

# In[15]:


counts.most_common(5)


# #### 

# In[16]:


counts.most_common()[-4:]


# #### 

# In[17]:


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

# In[18]:


df.head()


# #### 

# In[19]:


df.iloc[10]  # <1>


# #### 

# In[20]:


docs_tokens = []
for doc in docs:
    docs_tokens.append([
        tok.text.lower() for tok in nlp(doc.text)])  # <1>
len(docs_tokens[0])


# #### 

# In[21]:


all_doc_tokens = []
for tokens in docs_tokens:
    all_doc_tokens.extend(tokens)
len(all_doc_tokens)


# #### 

# In[22]:


vocab = sorted(  # <1>
    set(all_doc_tokens))  # <2>
len(vocab)


# #### 

# In[23]:


len(all_doc_tokens) / len(vocab)  # <3>


# #### 

# In[24]:


vocab  # <1>


# #### 

# In[25]:


count_vectors = []


# #### 

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [doc.text for doc in docs]
vectorizer = CountVectorizer()
count_vectors = vectorizer.fit_transform(corpus)  # <1>
print(count_vectors.toarray()) # <2>


# #### 

# In[29]:


import numpy as np
v1 = np.array(list(range(5)))
v2 = pd.Series(reversed(range(5)))
slow_answer = sum([4.2 * (x1 * x2) for x1, x2 in zip(v1, v2)])
slow_answer


# #### 

# In[30]:


faster_answer = sum(4.2 * v1 * v2)  # <1>
faster_answer


# #### 

# In[31]:


fastest_answer = 4.2 * v1.dot(v2)  # <2>
fastest_answer


# #### 

# In[36]:


v1.dot(v2) == (np.linalg.norm(v1) * np.linalg.norm(v2))  # * np.cos(angle_between_v1_and_v2)


# In[56]:


docs


# In[58]:


vectorizer.vocabulary_


# In[37]:


cos_similarity_between_A_and_B = np.cos(angle_between_A_and_B) \
   = A.dot(B) / (np.linalg.norm(A) * np.linalg.norm(B))


# #### 

# In[38]:


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

# In[39]:


from sklearn.metrics.pairwise import cosine_similarity
vec1 = count_vectors[1,:]
vec2 = count_vectors[2,:]
cosine_similarity(vec1, vec2)


# #### .Cosine similarity

# In[40]:


import copy
question = "What is algorithmic bias?"
ngram_docs = copy.copy(docs)
ngram_docs.append(question)


# In[78]:





# #### .Cosine similarity

# In[79]:


question_vec = vectorizer.transform([new_sentence])
question_vec


# In[81]:


question_vec.toarray()


# In[92]:


vocab = list(zip(*sorted((i, tok) for tok, i in 
    vectorizer.vocabulary_.items())))[1]
pd.Series(question_vec.toarray()[0], index=vocab).head(8)


# #### .Cosine similarity

# In[87]:


docs[3]


# In[98]:


cosine_similarity(count_vectors[1,:], count_vectors[3,:])


# In[53]:


cosine_similarity(count_vectors, new_sentence_vector)


# #### .Cosine similarity

# In[89]:


ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
ngram_vectors = ngram_vectorizer.fit_transform(corpus)
ngram_vectors


# In[96]:


vocab = list(zip(*sorted((i, tok) for tok, i in
    ngram_vectorizer.vocabulary_.items())))[1]
pd.DataFrame(ngram_vectors.toarray(), columns=vocab)['algorithmic bias']


# #### .Cosine similarity

# In[47]:


cosine_similarity(ngram_vectors[1,:], ngram_vectors[2,:])


# In[97]:


cosine_similarity(ngram_vectors[1,:], ngram_vectors[3,:])


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

