>>> import spacy
>>> spacy.cli.download("en_core_web_sm")
>>> nlp = spacy.load("en_core_web_sm")
>>> sentence = ('It has also arisen in criminal justice, healthcare, and '
...     'hiring, compounding existing racial, economic, and gender biases.')
>>> doc = nlp(sentence)
>>> tokens = [token.text for token in doc]
>>> tokens
bag_of_words.keys()
>>> from collections import Counter
>>> bag_of_words = Counter(tokens)
>>> bag_of_words
bag_of_words.most_common(2)
bag_of_words.most_common(3)
hist -o -p
bag_of_words.most_common()
pd.Series(bag_of_words.most_common(3))
import pandas as pd
pd.Series(bag_of_words.most_common(3))
pd.Series(dict(bag_of_words.most_common(3)))
pd.Series(dict(bag_of_words.most_common))
pd.Series(dict(bag_of_words.most_common()))
counts = pd.Series(dict(bag_of_words.most_common()))
counts
hist -o -p
counts / counts.sum()
>>> len(tokens)
counts['justice'] / counts.sum()
hist -o -p
counts['justice']
import wikipedia
import wikipedia
help(wikipedia)
>>> import requests
>>> DATA_DIR = ('https://gitlab.com/tangibleai/nlpia2/'
...             '-/raw/master/src/nlpia2/data')
>>> url = DATA_DIR + '/bias_intro.txt'
>>> bias_intro = requests.get(url).content.decode() # <1>
>>> bias_intro[:60]
url
>>> import requests
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/tree/master/rc/nlpia2/ch03/bias_intro.txt')
>>> bias_intro = requests.get(url).content.decode()  # <1>
>>> bias_intro[:60]
bias_intro
>>> import requests
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/tree/master/rc/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response
>>> import requests
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/tree/master/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response
url
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/tree/main/src/nlpia2/ch03/bias_intro.txt')
>>> response = requests.get(url)
>>> response
hist -o -p
url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/tree/main/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response
>>> bias_intro = response.content.decode()  # <1>
>>> bias_intro[:60]
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/raw/main/src/nlpia2/ch03/bias_intro.txt')
url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/tree/main/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response
url = ('https://gitlab.com/tangibleai/nlpia2/'
       '-/raw/main/src/nlpia2/ch03/bias_intro.txt')
response = requests.get(url)
response
response.content
response.content.decode()
>>> import requests
>>> url = ('https://gitlab.com/tangibleai/nlpia2/'
...        '-/raw/main/src/nlpia2/ch03/bias_intro.txt')
>>> response = requests.get(url)
>>> response

>>> bias_intro = response.content.decode()  # <1>
>>> bias_intro[:60]
>>> tokens = [tok.text for tok in nlp(bias_intro)]
>>> counts = Counter(tokens)
>>> counts
>>> counts.most_frequent(4)
>>> counts.most_common(4)
>>> counts.most_common(5)
hist -o -p
counts.least_common(5)
reversed(counts.most_common(5))
list(reversed(counts.most_common(5)))
counts.most_common()[-5]
counts.most_common()[-5:]
docs = list(nlp(bias_intro).sents)
counts = [Counter([t.text for t in s]) for s in docs]
pd.DataFrame(counts)
pd.DataFrame(counts).astype(int)
pd.DataFrame(counts).fillna(0).astype(int)
hist -o -p
pd.options.display.max_columns = 8
df = pd.DataFrame(counts).fillna(0).astype(int)
df
pd.options.display.max_columns = 5
df
pd.options.display.max_columns = 6
df
hist -o -p
docs = list(nlp(bias_intro).sents.lower())
counts = [Counter([t.text.lower() for t in s]) for s in docs]
counts = [Counter([t.text.lower() for t in s]) for s in docs]
df = pd.DataFrame(counts).fillna(0).astype(int)
df
hist -o -p
df.loc[3]
df.loc[3]['justice']
df.loc[4]['justice']
df.loc[2]['justice']
df['justice']
df.loc[10]['justice']
df.loc[10]
hist -o -p
Knowledge and Society in Times of Upheaval
import jieba
hist -o -p -f algorithmic_bias_examples_16_sents.ipy
hist -f algorithmic_bias_examples_16_sents.py
