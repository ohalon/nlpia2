import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load("en_core_web_sm")

DATA_DIR = 'https://gitlab.com/tangibleai/nlpia2/-/raw/' \
           'main/.nlpia2-data/manuscript/adoc/'

url = DATA_DIR + 'Chapter 05 -- Word Brain (artificial neural networks for NLP).adoc'

model = SentenceTransformer('msmarco-distilbert-cos-v5')

req = requests.get(url)

sents = []

for line in req.text.split('\n'):
    doc=nlp(line)
    sents.append(list(doc.sents))












