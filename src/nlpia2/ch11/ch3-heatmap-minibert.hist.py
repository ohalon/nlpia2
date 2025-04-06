df = update_nlpia_lines()
from nlpia2.text_processing.extractors import *
df = update_nlpia_lines()
df
df
df.sample(10).T
df3 = df[df["filename"] == "Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc"]
texts = df3.text[df3.is_text | df3.is_title]
nlp = spacy.load("en_core_web_md")
embeddings = texts.apply(lambda s: nlp(s).vector)
dfe = pd.DataFrame([list(x / np.linalg.norm(x)) for x in embeddings])
heatmap = dfe.values.dot(dfe.values.T)
heatmap.shape
imoprt spacy
>>> import spacy
>>> nlp = spacy.load('en_core_web_md')
df3 = df[df["filename"] == "Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc"]
texts = df3.text[df3.is_text | df3.is_title]

embeddings = texts.apply(lambda s: nlp(s).vector)
dfe = pd.DataFrame([list(x / np.linalg.norm(x)) for x in embeddings])
heatmap = dfe.values.dot(dfe.values.T)
heatmap.shape
import numpy as np
dfe = pd.DataFrame([list(x / np.linalg.norm(x)) for x in embeddings])
heatmap = dfe.values.dot(dfe.values.T)
heatmap.shape
heatmap
import seaborn as sns
sns.heatmap(heatmap)
from matplotlib import pyplot as plt
plt.show()
dfe.shape
heatmap.shape
df.shape
df3.shape
labels = texts.index.values
labels
texts.iloc[labels]
texts.loc[labels]
texts.loc[labels].str[:14]
labels = list(texts.loc[labels].str[:14])
heatmap = pd.DataFrame(heatmap, columns=labels, index=labels)
heatmap
sns.heatmap(heatmap)
plt.show()
sns.heatmap?
sns.heatmap(heatmap)
plt.xticks(rotation=-70)
plt.show()
sns.heatmap(heatmap)
plt.xticks(rotation=-30)
plt.show()
plt.xticks(rotation=-60)
plt.show()
plt.xticks(rotation=-60)
sns.heatmap(heatmap)
plt.xticks(rotation=-60)
plt.show()
sns.heatmap(heatmap)
plt.xticks(rotation=70)
plt.show()
plt.xticks(rotation=70, ha='right')
sns.heatmap(heatmap)
plt.xticks(rotation=70, ha='right')
plt.show()
sns.heatmap(heatmap)
plt.xticks(rotation=-35, ha='left')
plt.show()
hsit
hist
update_nlpia_lines??
LINES_FILEPATH
pd.read_csv('https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2/data/nlpia_lines.csv')
pd.read_csv('https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2/data/nlpia_lines.csv', index_col=0)
>>> import pandas as pd
>>> url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/'
>>> url += 'src/nlpia2/data/nlpia_lines.csv'
>>> df = pd.read_csv(url, index_col=0)
>>> chapter = 3
>>> df3 = df[df["filename"].str.startswith(f'Chapter-{chapter:02d}')]
>>> texts = df3.text[df3.is_text | df3.is_title]
df3.shape
df3.filename
imoprt numpy as np
import numpy as np
np.linalg.norm(embeddings)
np.linalg.norm(embeddings).shape
embeddings.shape
np.linalg.norm(embeddings, axis=1).shape
embeddings.apply(np.linalg.norm)
sns.heatmap(heatmap)=
from transformers import pipeline

bert = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base")
from transformers import pipeline
vecs = bert(texts)
text
type(texts)
from transformers import pipeline
vecs = bert(texts.values)
from transformers import pipeline
vecs = bert(list(texts.values))
hist -p
df.columns
df['chapter'] == 3
(df['chapter'] == 3).sum()
>>> url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/'
>>> url += 'src/nlpia2/data/nlpia_lines.csv'
>>> df = pd.read_csv(url, index_col=0)
>>> df = df[df.chapter == 3].copy()
>>> df.sample(5)['text', 'is_text', 'is_title', 'num_sents']

>>> df.sample(5)[['text', 'is_text', 'is_title', 'num_sents']]
>>> df[['text', 'is_text', 'is_title', 'num_sents']]
hist -o -p
hist -o -p
>>> df3.shape
texts.shape
vecs.shape
vecs = np.array(vecs)
vecs.shape
pd.DataFrame(vecs)
vecs[0]
vecs[0][0]
vecs[0][0][0]
len(vecs[0])
len(vecs)
len(vecs[0][0])
len(vecs[1][0])
len(vecs[2][0])
len(vecs[3][0])
from sentence_transformers import SentenceTransformer
SentenceTransformer?
SentenceTransformer('all-MiniLM-L12-v2')
minibert = _
hist -o -p -f src/nlpia2/ch11/ch3-heatmap.hist.ipy
hist -o -p -f src/nlpia2/ch11/ch3-heatmap-minibert.hist.ipy
hist -f src/nlpia2/ch11/ch3-heatmap-minibert.hist.py
