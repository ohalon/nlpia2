#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-02_Tokens-of-thought-natural-language-words`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-02_Tokens-of-thought-natural-language-words.adoc)

# #### .Example quote from _The Book Thief_ split into tokens

# In[ ]:


text = ("Trust me, though, the words were on their way, and when "
        "they arrived, Liesel would hold them in her hands like "
        "the clouds, and she would wring them out, like the rain.")
tokens = text.split()  # <1>
tokens[:8]


# #### .Example quote from _The Book Thief_ split into tokens

# In[ ]:


import re
pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
texts = [text]
texts.append("There's no such thing as survival of the fittest. "
             "Survival of the most adequate, maybe.")
tokens = list(re.findall(pattern, texts[-1]))
tokens[:8]


# #### .Example quote from _The Book Thief_ split into tokens

# In[ ]:


tokens[8:16]


# #### .Example quote from _The Book Thief_ split into tokens

# In[ ]:


tokens[16:]


# #### 

# In[ ]:


import numpy as np
vocab = sorted(set(tokens))  # <1>
' '.join(vocab[:12])  # <2>


# #### 

# In[ ]:


num_tokens = len(tokens)
num_tokens


# #### 

# In[ ]:


vocab_size = len(vocab)
vocab_size


# #### 

# In[ ]:


import spacy  # <1>
spacy.cli.download('en_core_web_sm')  # <2>
nlp = spacy.load('en_core_web_sm')  # <3>
doc = nlp(texts[-1])
type(doc)


# #### 

# In[ ]:


tokens = [tok.text for tok in doc]
tokens[:9]


# #### 

# In[ ]:


tokens[9:17]


# #### 

# In[ ]:


from spacy import displacy
sentence = list(doc.sents)[0]  # <1>
svg = displacy.render(sentence, style="dep",
    jupyter=False)  # <2>
open('sentence_diagram.svg', 'w').write(svg)  # <3>
displacy.render(sentence, style="dep")  # <5>


# #### 

# In[ ]:


import requests
text = requests.get('https://proai.org/nlpia2-ch2.adoc').text
f'{round(len(text) / 10_000)}0k'  # <1>


# #### 

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_sm')
get_ipython().run_line_magic('timeit', 'nlp(text)  # <1>')


# #### 

# In[ ]:


f'{round(len(text) / 10_000)}0k'


# #### 

# In[ ]:


doc = nlp(text)
f'{round(len(list(doc)) / 10_000)}0k'


# #### 

# In[ ]:


f'{round(len(doc) / 1_000 / 4.67)}kWPS'  # <2>


# #### 

# In[ ]:


nlp.pipe_names  # <1>


# #### 

# In[ ]:


nlp = spacy.load('en_core_web_sm', disable=nlp.pipe_names)
get_ipython().run_line_magic('timeit', 'nlp(text)')


# #### 

# In[ ]:


import nltk
nltk.download('punkt')


# #### 

# In[ ]:


from nltk.tokenize import word_tokenize
get_ipython().run_line_magic('timeit', 'word_tokenize(text)')


# #### 

# In[ ]:


tokens = word_tokenize(text)
f'{round(len(tokens) / 10_000)}0k'


# #### 

# In[ ]:


pattern = r'\w+(?:\'\w+)?|[^\w\s]'
tokens = re.findall(pattern, text)  # <1>
f'{round(len(tokens) / 10_000)}0k'


# #### 

# In[ ]:


get_ipython().run_line_magic('timeit', 're.findall(pattern, text)')


# #### 

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='char')
vectorizer.fit(texts)


# #### 

# In[ ]:


bpevocab_list = [
   sorted((i, s) for s, i in vectorizer.vocabulary_.items())]
bpevocab_dict = dict(bpevocab_list[0])
list(bpevocab_dict.values())[:7]


# #### 

# In[ ]:


vectors = vectorizer.transform(texts)
df = pd.DataFrame(
    vectors.todense(), 
    columns=vectorizer.vocabulary_)
df.index = [t[:8] + '...' for t in texts]
df = df.T
df['total'] = df.T.sum()
df


# #### 

# In[ ]:


df.sort_values('total').tail()


# #### 

# In[ ]:


df['n'] = [len(tok) for tok in vectorizer.vocabulary_]
df[df['n'] > 1].sort_values('total').tail()


# #### 

# In[ ]:


hi_text = 'Hiking home now'
hi_text.startswith('Hi')


# #### 

# In[ ]:


pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
'Hi' in re.findall(pattern, hi_text)  # <2>


# #### 

# In[ ]:


'Hi' == re.findall(pattern, hi_text)[0]  # <3>


# #### 

# In[ ]:


import pandas as pd
onehot_vectors = np.zeros(
    (len(tokens), vocab_size), int)  # <1>
for i, tok in enumerate(tokens):
    if tok not in vocab:
        continue
    onehot_vectors[i, vocab.index(tok)] = 1  # <2>
df_onehot = pd.DataFrame(onehot_vectors, columns=vocab)
df_onehot.shape


# #### 

# In[ ]:


df_onehot.iloc[:,:8].replace(0, '')  # <3>


# #### 

# In[ ]:


import spacy  # <1>
from nlpia2.spacy_language_model import load  # <2>
nlp = load('en_core_web_sm')  # <3>
nlp


# #### 

# In[ ]:


doc = nlp(texts[-1])
type(doc)


# #### 

# In[ ]:


tokens = [tok.text for tok in doc]  # <1>
tokens[:9]  # <2>


# #### 

# In[ ]:


tokens[9:17]


# #### 

# In[ ]:


from spacy import displacy
sentence = list(doc.sents)[0] # <1>
displacy.serve(sentence, style="dep")
get_ipython().system('firefox 127.0.0.1:5000')


# #### 

# In[ ]:


import requests
text = requests.get('https://proai.org/nlpia2-ch2.adoc').text
f'{round(len(text) / 10_000)}0k'  # <1>


# #### 

# In[ ]:


from nlpia2.spacy_language_model import load
nlp = load('en_core_web_sm')
get_ipython().run_line_magic('timeit', 'nlp(text)  # <1>')


# #### 

# In[ ]:


f'{round(len(text) / 10_000)}0k'


# #### 

# In[ ]:


doc = nlp(text)
f'{round(len(list(doc)) / 10_000)}0k'


# #### 

# In[ ]:


f'{round(len(doc) / 1_000 / 4.67)}kWPS'  # <2>


# #### 

# In[ ]:


nlp.pipe_names  # <1>


# #### 

# In[ ]:


nlp = load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser'])
nlp.pipe_names


# #### 

# In[ ]:


get_ipython().run_line_magic('timeit', 'nlp(text)')


# #### 

# In[ ]:


import nltk


# #### 

# In[ ]:


pattern = r'\w+(?:\'\w+)?|[^\w\s]'
tokens = re.findall(pattern, text)  # <1>
f'{round(len(tokens) / 10_000)}0k'


# #### 

# In[ ]:


get_ipython().run_line_magic('timeit', 're.findall(pattern, text)')


# #### 

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='char')
vectorizer.fit(texts)


# #### 

# In[ ]:


bpevocab_list = [
   sorted((i, s) for s, i in vectorizer.vocabulary_.items())]
bpevocab_dict = dict(bpevocab_list[0])
list(bpevocab_dict.values())[:7]


# #### 

# In[ ]:


vectors = vectorizer.transform(texts)
df = pd.DataFrame(
    vectors.todense(),
    columns=vectorizer.vocabulary_)
df.index = [t[:8] + '...' for t in texts]
df = df.T
df['total'] = df.T.sum()
df


# #### 

# In[ ]:


df.sort_values('total').tail(3)


# #### 

# In[ ]:


df['n'] = [len(tok) for tok in vectorizer.vocabulary_]
df[df['n'] > 1].sort_values('total').tail()


# #### 

# In[ ]:


hi_text = 'Hiking home now'
hi_text.startswith('Hi')


# #### 

# In[ ]:


pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
'Hi' in re.findall(pattern, hi_text)  # <2>


# #### 

# In[ ]:


'Hi' == re.findall(pattern, hi_text)[0]  # <3>


# #### 

# In[ ]:


bow = sorted(set(re.findall(pattern, text)))
bow[:9]


# #### 

# In[ ]:


bow[9:19]


# #### 

# In[ ]:


bow[19:27]


# #### .Example dot product calculation

# In[ ]:


v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])
v1.dot(v2)


# #### .Example dot product calculation

# In[ ]:


(v1 * v2).sum()  # <1>


# #### .Example dot product calculation

# In[ ]:


sum([x1 * x2 for x1, x2 in zip(v1, v2)])  # <2>


# #### .Example dot product calculation

# In[ ]:


from nltk.tokenize import TreebankWordTokenizer
texts.append(
  "If conscience and empathy were impediments to the advancement of "
  "self-interest, then we would have evolved to be amoral sociopaths."
  )  # <1>
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(texts[-1])[:6]
tokens[:8]


# #### .Example dot product calculation

# In[ ]:


tokens[8:16]


# #### .Example dot product calculation

# In[ ]:


tokens[16:]


# #### 

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")
text = "Nice guys finish first."  # <1>
doc = nlp(text)
for token in doc:
    print(f"{token.text:<11}{token.pos_:<10}{token.dep:<10}")


# #### 

# In[ ]:


import jieba
seg_list = jieba.cut("西安是一座举世闻名的文化古城")  # <1>
list(seg_list)


# #### 

# In[ ]:


import jieba
seg_list = jieba.cut("西安是一座举世闻名的文化古城", cut_all=True)  # <1>
list(seg_list)


# #### 

# In[ ]:


import jieba
from jieba import posseg
words = posseg.cut("西安是一座举世闻名的文化古城")
jieba.enable_paddle()  # <1>
words = posseg.cut("西安是一座举世闻名的文化古城", use_paddle=True)
list(words)


# #### 

# In[ ]:


import spacy
spacy.cli.download("zh_core_web_sm")  # <1>
nlpzh = spacy.load("zh_core_web_sm")
doc = nlpzh("西安是一座举世闻名的文化古城")
[(tok.text, tok.pos_) for tok in doc]


# #### 

# In[ ]:


from nltk.tokenize.casual import casual_tokenize
texts.append("@rickrau mind BLOOOOOOOOWWWWWN by latest lex :*) !!!!!!!!")
casual_tokenize(texts[-1], reduce_len=True)


# #### .Broad list of stop words

# In[ ]:


import requests
url = ("https://gitlab.com/tangibleai/nlpia/-/raw/master/"
       "src/nlpia/data/stopword_lists.json")
response = requests.get(url)
stopwords = response.json()['exhaustive']  # <1>
tokens = 'the words were just as I remembered them'.split()  # <2>
tokens_without_stopwords = [x for x in tokens if x not in stopwords]
print(tokens_without_stopwords)


# #### .Broad list of stop words

# In[ ]:


import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
len(stop_words)


# #### .Broad list of stop words

# In[ ]:


stop_words[:7]


# #### .Broad list of stop words

# In[ ]:


[sw for sw in stopwords if len(sw) == 1]


# #### .Broad list of stop words

# In[ ]:


resp = requests.get(url)


# #### 

# In[ ]:


tokens = ['House', 'Visitor', 'Center']
normalized_tokens = [x.lower() for x in tokens]
print(normalized_tokens)


# #### 

# In[ ]:


def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$',
        word)[0][0].strip("'") for word in phrase.lower().split()])
stem('houses')


# #### 

# In[ ]:


stem("Doctor House's calls")


# #### 

# In[ ]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
' '.join([stemmer.stem(w).strip("'") for w in
  "dish washer's fairly washed dishes".split()])


# #### 

# In[ ]:


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
' '.join([stemmer.stem(w).strip("'") for w in
  "dish washer's fairly washed dishes".split()])


# #### 

# In[ ]:


nltk.download('wordnet')


# #### 

# In[ ]:


nltk.download('omw-1.4')


# #### 

# In[ ]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("better")  # <1>


# #### 

# In[ ]:


lemmatizer.lemmatize("better", pos="a")  # <2>


# #### 

# In[ ]:


lemmatizer.lemmatize("good", pos="a")


# #### 

# In[ ]:


stemmer.stem('goodness')


# #### 

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("better good goods goodness best")
for token in doc:
    print(token.text, token.lemma_)


# #### 

# In[ ]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()
sa.lexicon  # <1>


# #### 

# In[ ]:


[(tok, score) for tok, score in sa.lexicon.items()
  if " " in tok]  # <4>


# #### 

# In[ ]:


sa.polarity_scores(text=\
  "Python is very readable and it's great for NLP.")


# #### 

# In[ ]:


sa.polarity_scores(text=\
  "Python is not a bad choice for most applications.")


# #### 

# In[ ]:


corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print('{:+}: {}'.format(scores['compound'], doc))


# #### 

# In[ ]:


movies = pd.read_csv('https://proai.org/movie-reviews.csv.gz',
    index_col=0)
movies.head().round(2)


# #### 

# In[ ]:


movies.describe().round(2)


# #### 

# In[ ]:


import pandas as pd
pd.options.display.width = 75  # <1>
from nltk.tokenize import casual_tokenize  # <2>
bows = []
from collections import Counter  # <3>
for text in movies.text:
    bows.append(Counter(casual_tokenize(text)))
df_movies = pd.DataFrame.from_records(bows)  # <4>
df_movies = df_movies.fillna(0).astype(int)  # <5>
df_movies.shape  # <6>


# #### 

# In[ ]:


df_movies.head()


# #### 

# In[ ]:


df_movies.head()[list(bows[0].keys())]


# #### 

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb = nb.fit(df_movies, movies.sentiment > 0)  # <1>
movies['pred_senti'] = (
  nb.predict_proba(df_movies))[:, 1] * 8 - 4  # <2>
movies['error'] = movies.pred_senti - movies.sentiment
mae = movies['error'].abs().mean().round(1)  # <3>
mae


# #### 

# In[ ]:


movies['senti_ispos'] = (movies['sentiment'] > 0).astype(int)
movies['pred_ispos'] = (movies['pred_senti'] > 0).astype(int)
columns = [c for c in movies.columns if 'senti' in c or 'pred' in c]
movies[columns].head(8)


# #### 

# In[ ]:


(movies.pred_ispos ==
  movies.senti_ispos).sum() / len(movies)


# #### 

# In[ ]:


products = pd.read_csv('https://proai.org/product-reviews.csv.gz')
products.columns


# #### 

# In[ ]:


products.head()


# #### 

# In[ ]:


bows = []
for text in products['text']:
    bows.append(Counter(casual_tokenize(text)))
df_products = pd.DataFrame.from_records(bows)
df_products = df_products.fillna(0).astype(int)
df_products.shape # <1>


# #### 

# In[ ]:


df_all_bows = pd.concat([df_movies, df_products])
df_all_bows.columns  # <1>


# #### 

# In[ ]:


vocab = list(df_movies.columns)  # <1>
df_products = df_all_bows.iloc[len(movies):]  # <2>
df_products = df_products[vocab]  # <3>
df_products.shape


# #### 

# In[ ]:


df_movies.shape  # <4>


# #### 

# In[ ]:


products['senti_ispos'] = (products['sentiment'] > 0).astype(int)
products['pred_ispos'] = nb.predict(df_products).astype(int)
correct = (products['pred_ispos']
        == products['senti_ispos'])  # <1>
correct.sum() / len(products)

