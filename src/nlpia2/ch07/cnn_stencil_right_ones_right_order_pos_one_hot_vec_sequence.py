import spacy
# https://spacy.io
import spacy
nlp = 'en_core_web_md'
try:
    nlp = spacy.load(nlp)
except OSError:
    spacy.cli.download(nlp)
nlp = spacy.load(nlp) if isinstance(nlp, str) else nlp
text = 'right ones in the right order you can nudge the world'
doc = nlp(text)
import pandas as pd
df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()}])
df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()} for t in doc])
df
hist
pd.get_dummies(df, columns=['pos_'])
df = pd.get_dummies(df, columns={'pos_': ''})
df
df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()} for t in doc])
pd.get_dummies?
pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')
df = pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')
df
df.T
pwd
hist -o -p -f ../nlpia2/src/nlpia2/ch07/cnn-stencil-right-ones-right-order-pos-one-hot-vec-sequence.md
hist -f ../nlpia2/src/nlpia2/ch07/cnn_stencil_right_ones_right_order_pos_one_hot_vec_sequence.py
