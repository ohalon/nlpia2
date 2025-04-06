text = ("Trust me, though, the words were on their way, and when "
        "they arrived, Liesel would hold them in her hands like "
        "the clouds, and she would wring them out, like the rain.")

tokens = text.split()

tokens[:8]

import re

pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>

texts = [text]

texts.append("There's no such thing as survival of the fittest. "
             "Survival of the most adequate, maybe.")

tokens = list(re.findall(pattern, texts[-1]))

tokens[:8]

tokens[8:16]

tokens[16:]

import numpy as np  # <1>

vocab = sorted(set(tokens))  # <2>

' '.join(vocab[:12])  # <3>

num_tokens = len(tokens)

num_tokens

vocab_size = len(vocab)

vocab_size

import pandas as pd

onehot_vectors = np.zeros(
    (len(tokens), vocab_size), int)  # <1>

for i, tok in enumerate(tokens):
    if tok not in vocab:
        continue
    onehot_vectors[i, vocab.index(tok)] = 1  # <2>

df_onehot = pd.DataFrame(onehot_vectors, columns=vocab)

df_onehot.shape

df_onehot.iloc[:,:8].replace(0, '')  # <3>

import spacy  # <1>

from nlpia2.spacy_language_model import load  # <2>

nlp = load('en_core_web_sm')  # <3>

nlp

doc = nlp(texts[-1])

type(doc)

tokens = [tok.text for tok in doc]  # <1>

tokens[:9]  # <2>

tokens[9:17]

from spacy import displacy

sentence = list(doc.sents)[0] # <1>

displacy.serve(sentence, style="dep")

!firefox 127.0.0.1:5000

import requests

text = requests.get('https://proai.org/nlpia2-ch2.adoc').text

f'{round(len(text) / 10_000)}0k'  # <1>

from nlpia2.spacy_language_model import load

nlp = load('en_core_web_sm')

%timeit nlp(text)  # <1>

f'{round(len(text) / 10_000)}0k'

doc = nlp(text)

f'{round(len(list(doc)) / 10_000)}0k'

f'{round(len(doc) / 1_000 / 4.67)}kWPS'  # <2>

nlp.pipe_names  # <1>

nlp = load('en_core_web_sm', disable=['tok2vec', 'tagger', 'parser'])

nlp.pipe_names

%timeit nlp(text)

import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

%timeit word_tokenize(text)

tokens = word_tokenize(text)

f'{round(len(tokens) / 10_000)}0k'

pattern = r'\w+(?:\'\w+)?|[^\w\s]'

tokens = re.findall(pattern, text)  # <1>

f'{round(len(tokens) / 10_000)}0k'

%timeit re.findall(pattern, text)

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='char')

vectorizer.fit(texts)

bpevocab_list = [
   sorted((i, s) for s, i in vectorizer.vocabulary_.items())]

bpevocab_dict = dict(bpevocab_list[0])

list(bpevocab_dict.values())[:7]

vectors = vectorizer.transform(texts)

df = pd.DataFrame(
    vectors.todense(), 
    columns=vectorizer.vocabulary_)

df.index = [t[:8] + '...' for t in texts]

df = df.T

df['total'] = df.T.sum()

df

df.sort_values('total').tail(3)

df['n'] = [len(tok) for tok in vectorizer.vocabulary_]

df[df['n'] > 1].sort_values('total').tail()

hi_text = 'Hiking home now'

hi_text.startswith('Hi')

pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>

'Hi' in re.findall(pattern, hi_text)  # <2>

'Hi' == re.findall(pattern, hi_text)[0]  # <3>

bow = sorted(set(re.findall(pattern, text)))

bow[:9]

bow[9:19]

bow[19:27]

v1 = pd.np.array([1, 2, 3])

v2 = pd.np.array([2, 3, 4])

v1.dot(v2)

(v1 * v2).sum()  # <1>

sum([x1 * x2 for x1, x2 in zip(v1, v2)])  # <2>

from nltk.tokenize import TreebankWordTokenizer

texts.append(
  "If conscience and empathy were impediments to the advancement of "
  "self-interest, then we would have evolved to be amoral sociopaths."
  )  # <1>

tokenizer = TreebankWordTokenizer()

tokens = tokenizer.tokenize(texts[-1])[:6]

tokens[:8]

tokens[8:16]

tokens[16:]

import spacy

nlp = spacy.load("en_core_web_sm")

text = "Nice guys finish first."  # <1>

doc = nlp(text)

for token in doc:

    print(f"{token.text:<11}{token.pos_:<10}{token.dep:<10}")

seg_list = jieba.cut("西安是一座举世闻名的文化古城")  # <1>

list(seg_list)

import jieba
seg_list = jieba.cut("西安是一座举世闻名的文化古城", cut_all=True)  # <1>

list(seg_list)

seg_list = jieba.cut_for_search("西安是一座举世闻名的文化古城")  # <1>

list(seg_list)

import jieba

from jieba import posseg

words = posseg.cut("西安是一座举世闻名的文化古城")

jieba.enable_paddle()  # <1>

words = posseg.cut("西安是一座举世闻名的文化古城", use_paddle=True)

list(words)

import spacy

spacy.cli.download("zh_core_web_sm")  # <1>

nlpzh = spacy.load("zh_core_web_sm")

doc = nlpzh("西安是一座举世闻名的文化古城")

[(tok.text, tok.pos_) for tok in doc]

from nltk.tokenize.casual import casual_tokenize

texts.append("@rickrau mind BLOOOOOOOOWWWWWN by latest lex :*) !!!!!!!!")

casual_tokenize(texts[-1], reduce_len=True)

import requests

url = ("https://gitlab.com/tangibleai/nlpia/-/raw/master/"
       "src/nlpia/data/stopword_lists.json")

response = requests.get(url)

stopwords = response.json()['exhaustive']  # <1>

tokens = 'the words were just as I remembered them'.split()  # <2>

tokens_without_stopwords = [x for x in tokens if x not in stopwords]

print(tokens_without_stopwords)

import nltk

nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')

len(stop_words)

stop_words[:7]

[sw for sw in stopwords if len(sw) == 1]

resp = requests.get(url)

len(resp.json()['exhaustive'])

len(resp.json()['sklearn'])

len(resp.json()['spacy'])

len(resp.json()['nltk'])

len(resp.json()['reuters'])

tokens = ['House', 'Visitor', 'Center']

normalized_tokens = [x.lower() for x in tokens]

print(normalized_tokens)

def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$',
        word)[0][0].strip("'") for word in phrase.lower().split()])

stem('houses')

stem("Doctor House's calls")

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

' '.join([stemmer.stem(w).strip("'") for w in
  "dish washer's fairly washed dishes".split()])

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')

' '.join([stemmer.stem(w).strip("'") for w in
  "dish washer's fairly washed dishes".split()])

nltk.download('wordnet')

nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize("better")  # <1>

lemmatizer.lemmatize("better", pos="a")  # <2>

lemmatizer.lemmatize("good", pos="a")

lemmatizer.lemmatize("goods", pos="a")

lemmatizer.lemmatize("goods", pos="n")

lemmatizer.lemmatize("goodness", pos="n")

lemmatizer.lemmatize("best", pos="a")

stemmer.stem('goodness')

import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("better good goods goodness best")

for token in doc:

    print(token.text, token.lemma_)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sa = SentimentIntensityAnalyzer()

sa.lexicon  # <1>

[(tok, score) for tok, score in sa.lexicon.items()
  if " " in tok]  # <4>

sa.polarity_scores(text=\
  "Python is very readable and it's great for NLP.")

sa.polarity_scores(text=\
  "Python is not a bad choice for most applications.")

corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]

for doc in corpus:
    scores = sa.polarity_scores(doc)
    print('{:+}: {}'.format(scores['compound'], doc))

movies = pd.read_csv('https://proai.org/movie-reviews.csv.gz', \
    index_col=0)

movies.head().round(2)

movies.describe().round(2)

import pandas as pd

pd.options.display.width = 75  # <1>

from nltk.tokenize import casual_tokenize  # <2>

bags_of_words = []

from collections import Counter  # <3>

for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))

df_bows = pd.DataFrame.from_records(bags_of_words)  # <4>

df_bows = df_bows.fillna(0).astype(int)  # <5>

df_bows.shape  # <6>

df_bows.head()

df_bows.head()[list(bags_of_words[0].keys())]

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb = nb.fit(df_bows, movies.sentiment > 0)  # <1>

movies['pred_senti'] = (
  nb.predict_proba(df_bows))[:, 1] * 8 - 4  # <2>

movies['error'] = movies.pred_senti - movies.sentiment

mae = movies['error'].abs().mean().round(1)  # <3>

mae

movies['senti_ispos'] = (movies['sentiment'] > 0).astype(int)

movies['pred_ispos'] = (movies['pred_senti'] > 0).astype(int)

columns = [c for c in movies.columns if 'senti' in c or 'pred' in c]

movies[columns].head(8)

(movies.pred_ispos ==
  movies.senti_ispos).sum() / len(movies)

products = pd.read_csv('https://proai.org/product-reviews.csv.gz')

for text in products['text']:
    bags_of_words.append(Counter(casual_tokenize(text)))

df_product_bows = pd.DataFrame.from_records(bags_of_words)

df_product_bows = df_product_bows.fillna(0).astype(int)

df_all_bows = df_bows.append(df_product_bows)

df_all_bows.columns  # <1>

df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]  # <2>

df_product_bows.shape

df_bows.shape  # <3>

products['senti_ispos'] = (products['sentiment'] > 0).astype(int)

products['pred_ispos'] = nb.predict(df_product_bows).astype(int)

products.head()

tp = products['pred_ispos'] == products['senti_ispos']  # <1>

tp.sum() / len(products)
