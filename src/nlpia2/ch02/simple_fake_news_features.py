import pandas as pd
import re
from pathlib import Path


DATA_DIR = Path('~/code/tangibleai/fake-news/data').expanduser().resolve().absolute()
df = pd.read_csv(Path(DATA_DIR) / 'all.csv.gz', index_col=None)

pattern = r'\w+(?:\'\w+)?|[^\w\s]'
df['title_len'] = df['title'].str.len()
df['title_num_tokens'] = df['title'].str.findall(pattern).apply(len)
df['title_ave_token_len'] = df['title_len'] / df['title_num_tokens']

wpat = r'\w+(?:\'\w+)?'
df['title_num_words'] = df['title'].str.findall(wpat).str.len()
df['title_wlen'] = df['title'].str.findall(wpat).str.join('').str.len()
df['title_ave_wordlen'] = df['title_wlen'] / df['title_num_words']

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
df = df.sample(len(df))  # <1>
# <1> Shuffle your data to improve convergence when your data is "stratified" by the target as it was here.

X = df[[c for c in df.columns if c.startswith('title_')]]
y = df['isfake']


def tokenize(s):
    pattern = r'\w+(?:\'\w+)?|[^\w\s]'
    return re.findall(pattern)


def tokenize_words(s):
    wordpat = r'\w+(?:\'\w+)?'
    return re.findall(wordpat)


def filter_nonwords(tokens):
    return [t for t in tokens if t and t[0]]


def _wordlen(tokens):
    return tokenize_words()
