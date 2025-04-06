import spacy

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

sentence = ('It has also arisen in criminal justice, healthcare, and '
    'hiring, compounding existing racial, economic, and gender biases.')

doc = nlp(sentence)

tokens = [token.text for token in doc]

tokens

from collections import Counter

bag_of_words = Counter(tokens)

bag_of_words

bag_of_words.most_common(3)  # <2>

counts = pd.Series(dict(bag_of_words.most_common()))  # <1>

counts

len(tokens)

counts.sum()

counts / counts.sum()  # <3>

counts['justice']

counts['justice'] / counts.sum()

sentence = "Algorithmic bias has been cited in cases ranging from " \
    "election outcomes to the spread of online hate speech."

tokens = [tok.text for tok in nlp(sentence)]

counts = Counter(tokens)

counts

import requests

url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/raw/main/src/nlpia2/ch03/bias_intro.txt')

response = requests.get(url)

response

bias_intro_bytes = response.content  # <1>

bias_intro = response.text  # <2>

assert bias_intro_bytes.decode() == bias_intro    # <3>

bias_intro[:60]

tokens = [tok.text for tok in nlp(bias_intro)]

counts = Counter(tokens)

counts

counts.most_common(5)

counts.most_common()[-4:]

counts.most_common()[-4:]

docs = [nlp(s) for s in bias_intro.split('\n') if s.strip()]  # <1>

counts = []

for doc in docs:
    counts.append(Counter([t.text.lower() for t in doc]))  # <2>

df = pd.DataFrame(counts)

df = df.fillna(0).astype(int)  # <3>

df.head()

df.loc[10]  # <1>

docs = list(nlp(bias_intro).sents)

counts = []

for doc in docs:
    counts.append(Counter([t.text.lower() for t in doc]))

df = pd.DataFrame(counts)

df = df.fillna(0).astype(int)  # <1>

df

docs_tokens = []

for doc in docs:
    doc_text = doc.text.lower()  # <1>
    docs_tokens.append([tok.text for tok in nlp(doc_text)])

len(docs_tokens[0])

all_doc_tokens = []

for doc_tokens in docs_tokens:
    all_doc_tokens.extend(doc_tokens)

len(all_doc_tokens)

vocab = sorted(set(all_doc_tokens))

len(vocab)

vocab

from collections import OrderedDict

zero_vector = OrderedDict((token, 0) for token in lexicon)

list(zero_vector.items())[:10]  # <1>

import copy

doc_vectors = []

for doc in docs:
    vec = copy.copy(zero_vector)  # <1>
    tokens = [token.text for token in nlp(doc.lower())]
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

from sklearn.feature_extraction.text import CountVectorizer

corpus = [doc.text for doc in docs]

vectorizer = CountVectorizer()

count_vectors = vectorizer.fit_transform(corpus)  # <1>

print(count_vectors.toarray()) # <2>

v1 = np.array(list(range(5)))

v2 = pd.Series(reversed(range(5)))

slow_answer = sum([4.2 * (x1 * x2) for x1, x2 in zip(v1, v2)])

slow_answer

faster_answer = sum(4.2 * v1 * v2)  # <1>

faster_answer

fastest_answer = 4.2 * v1.dot(v2)  # <2>

fastest_answer

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

from sklearn.metrics.pairwise import cosine_similarity

vec1 = count_vectors[1,:]

vec2 = count_vectors[2,:]

cosine_similarity(vec1, vec2)

new_sentence = "What is algorithmic bias?"

ngram_docs = copy.copy(docs)

ngram_docs.append(new_sentence)

new_sentence_vector = vectorizer.transform([new_sentence])

print(new_sentence_vector.toarray())

cosine_similarity(count_vectors[1,:], new_sentence)

ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))

ngram_vectors = ngram_vectorizer.fit_transform(corpus)

print(ngram_vectors.toarray())

cosine_similarity(ngram_vectors[1,:], ngram_vectors[2,:])

from this import s

print (s)

char_vectorizer = CountVectorizer(
    ngram_range=(1,1), analyzer='char')  # <1>

s_char_frequencies = char_vectorizer.fit_transform(s)

generate_histogram(
    s_char_frequencies, s_char_vectorizer)  # <2>

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
            '-/raw/master/src/nlpia/data')

url = DATA_DIR + '/machine_learning_full_article.txt'

ml_text = requests.get(url).content.decode()

ml_char_frequencies = char_vectorizer.fit_transform(ml_text)

generate_histogram(s_char_frequencies, s_char_vectorizer)

peak_distance = ord('R') - ord('E')

peak_distance

chr(ord('v') - peak_distance)  # <1>

chr(ord('n') - peak_distance)  # <2>

chr(ord('W') - peak_distance)

import codecs

print(codecs.decode(s, 'rot-13'))

nltk.download('brown')  # <1>

from nltk.corpus import brown

brown.words()[:10]  # <2>

brown.tagged_words()[:5]  # <3>

len(brown.words())

from collections import Counter

puncs = set((',', '.', '--', '-', '!', '?',
    ':', ';', '``', "''", '(', ')', '[', ']'))

word_list = (x.lower() for x in brown.words() if x not in puncs)

token_counts = Counter(word_list)

token_counts.most_common(10)

DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
            '-/raw/master/src/nlpia/data')

url = DATA_DIR + '/bias_discrimination.txt'

bias_discrimination = requests.get(url).content.decode()

intro_tokens = [token.text for token in nlp(bias_intro.lower())]

disc_tokens = [token.text for token in nlp(bias_discrimination.lower())]

intro_total = len(intro_tokens)

intro_total

disc_total = len (disc_tokens)

disc_total

intro_tf = {}

disc_tf = {}

intro_counts = Counter(intro_tokens)

intro_tf['bias'] = intro_counts['bias'] / intro_total

disc_counts = Counter(disc_tokens)

disc_tf['bias'] = disc_counts['bias'] / disc_total

'Term Frequency of "bias" in intro is:{:.4f}'.format(intro_tf['bias'])

'Term Frequency of "bias" in discrimination chapter is: {:.4f}'\
    .format(disc_tf['bias'])

intro_tf['and'] = intro_counts['and'] / intro_total

disc_tf['and'] = disc_counts['and'] / disc_total

print('Term Frequency of "and" in intro is: {:.4f}'\
    .format(intro_tf['and']))

print('Term Frequency of "and" in discrimination chapter is: {:.4f}'\
    .format(disc_tf['and']))

num_docs_containing_and = 0

for doc in [intro_tokens, disc_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1  # <1>

intro_tf['black'] = intro_counts['black'] / intro_total

disc_tf['black'] = disc_counts['black'] / disc_total

num_docs = 2

intro_idf = {}

disc_idf = {}

intro_idf['and'] = num_docs / num_docs_containing_and

disc_idf['and'] = num_docs / num_docs_containing_and

intro_idf['bias'] = num_docs / num_docs_containing_bias

disc_idf['bias'] = num_docs / num_docs_containing_bias

intro_idf['black'] = num_docs / num_docs_containing_black

disc_idf['black'] = num_docs / num_docs_containing_black

intro_tfidf = {}

intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']

intro_tfidf['bias'] = intro_tf['bias'] * intro_idf['bias']

intro_tfidf['black'] = intro_tf['black'] * intro_idf['black']

disc_tfidf = {}

disc_tfidf['and'] = disc_tf['and'] * disc_idf['and']

disc_tfidf['bias'] = disc_tf['bias'] * disc_idf['bias']

disc_tfidf['black'] = disc_tf['black'] * disc_idf['black']

log_tf = log(term_occurences_in_doc) -\
    log(num_terms_in_doc)  # <1>

log_log_idf = log(log(total_num_docs) -\
    log(num_docs_containing_term))  # <2>

log_tf_idf = log_tf + log_log_idf  # <3>

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
        tf = value / len(lexicon)
        if docs_containing_key:
            idf = len(docs) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    doc_tfidf_vectors.append(vec)

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

cosine_sim(query_vec, doc_tfidf_vectors[1])

cosine_sim(query_vec, doc_tfidf_vectors[2])

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = docs

vectorizer = TfidfVectorizer(min_df=1) # <1>

vectorizer = vectorizer.fit(corpus)  # <2>

vectors = vectorizer.transform(corpus)  # <3>

print(vectors.todense().round(2))  # <4>

DS_FAQ_URL = ('https://gitlab.com/tangibleai/qary/-/raw/main/'

qa_dataset = pd.read_csv(DS_FAQ_URL)

vectorizer = TfidfVectorizer()

vectorizer.fit(df['question'])

tfidfvectors_sparse = vectorizer.transform(df['question']) #<1>

tfidfvectors = tfidfvectors_sparse.todense() #<2>

def bot_reply(question):
   question_vector = vectorizer.transform([question]).todense()
   idx = question_vector.dot(tfidfvectors.T).argmax() # <1>

   print(
       f"Your question:\n  {question}\n\n"
       f"Most similar FAQ question:\n  {df['question'][idx]}\n\n"
       f"Answer to that FAQ question:\n  {df['answer'][idx]}\n\n"
   )

bot_reply("What's overfitting a model?")

bot_reply('How do I decrease overfitting for Logistic Regression?')
