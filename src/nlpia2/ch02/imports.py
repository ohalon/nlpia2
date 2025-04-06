import re  # noqa
import numpy as np  # noqa
import spacy
from spacy import displacy  # noqa
import pandas as pd  # noqa
from nltk.tokenize import word_tokenize, TreebankWordTokenizer  # noqa
import jieba  # noqa

# spacy.cli.download('en_core_web_sm')  # <1>
nlp = spacy.load('en_core_web_sm')

# so tables are no wider than what the book can handle:
pd.options.display.width = 75 
# pd.set_option('display.width', 75)
