>>> import spacy
>>> # https://spacy.io
... import spacy
... nlp = 'en_core_web_md'
... try:
...     nlp = spacy.load(nlp)
... except OSError:
...     spacy.cli.download(nlp)
... nlp = spacy.load(nlp) if isinstance(nlp, str) else nlp
...
>>> text = 'right ones in the right order you can nudge the world'
>>> doc = nlp(text)
>>> import pandas as pd
>>> df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()}])
>>> df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()} for t in doc])
>>> df
     text  pos_
0   right   ADJ
1    ones  NOUN
2      in   ADP
3     the   DET
4   right   ADJ
5   order  NOUN
6     you  PRON
7     can   AUX
8   nudge  VERB
9     the   DET
10  world  NOUN
>>> hist
>>> pd.get_dummies(df, columns=['pos_'])
     text  pos__ADJ  pos__ADP  ...  pos__NOUN  pos__PRON  pos__VERB
0   right         1         0  ...          0          0          0
1    ones         0         0  ...          1          0          0
2      in         0         1  ...          0          0          0
3     the         0         0  ...          0          0          0
4   right         1         0  ...          0          0          0
5   order         0         0  ...          1          0          0
6     you         0         0  ...          0          1          0
7     can         0         0  ...          0          0          0
8   nudge         0         0  ...          0          0          1
9     the         0         0  ...          0          0          0
10  world         0         0  ...          1          0          0

[11 rows x 8 columns]
>>> df = pd.get_dummies(df, columns={'pos_': ''})
>>> df
     text  pos__ADJ  pos__ADP  ...  pos__NOUN  pos__PRON  pos__VERB
0   right         1         0  ...          0          0          0
1    ones         0         0  ...          1          0          0
2      in         0         1  ...          0          0          0
3     the         0         0  ...          0          0          0
4   right         1         0  ...          0          0          0
5   order         0         0  ...          1          0          0
6     you         0         0  ...          0          1          0
7     can         0         0  ...          0          0          0
8   nudge         0         0  ...          0          0          1
9     the         0         0  ...          0          0          0
10  world         0         0  ...          1          0          0

[11 rows x 8 columns]
>>> df = pd.DataFrame([{k: getattr(t, k) for k in 'text pos_'.split()} for t in doc])
>>> pd.get_dummies?
>>> pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')
     text  ADJ  ADP  AUX  DET  NOUN  PRON  VERB
0   right    1    0    0    0     0     0     0
1    ones    0    0    0    0     1     0     0
2      in    0    1    0    0     0     0     0
3     the    0    0    0    1     0     0     0
4   right    1    0    0    0     0     0     0
5   order    0    0    0    0     1     0     0
6     you    0    0    0    0     0     1     0
7     can    0    0    1    0     0     0     0
8   nudge    0    0    0    0     0     0     1
9     the    0    0    0    1     0     0     0
10  world    0    0    0    0     1     0     0
>>> df = pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')
>>> df
     text  ADJ  ADP  AUX  DET  NOUN  PRON  VERB
0   right    1    0    0    0     0     0     0
1    ones    0    0    0    0     1     0     0
2      in    0    1    0    0     0     0     0
3     the    0    0    0    1     0     0     0
4   right    1    0    0    0     0     0     0
5   order    0    0    0    0     1     0     0
6     you    0    0    0    0     0     1     0
7     can    0    0    1    0     0     0     0
8   nudge    0    0    0    0     0     0     1
9     the    0    0    0    1     0     0     0
10  world    0    0    0    0     1     0     0
>>> df.T
         0     1   2    3      4      5    6    7      8    9      10
text  right  ones  in  the  right  order  you  can  nudge  the  world
ADJ       1     0   0    0      1      0    0    0      0    0      0
ADP       0     0   1    0      0      0    0    0      0    0      0
AUX       0     0   0    0      0      0    0    1      0    0      0
DET       0     0   0    1      0      0    0    0      0    1      0
NOUN      0     1   0    0      0      1    0    0      0    0      1
PRON      0     0   0    0      0      0    1    0      0    0      0
VERB      0     0   0    0      0      0    0    0      1    0      0
>>> pwd
'/home/hobs/code/tangibleai/nlpia-manuscript'
>>> hist -o -p -f ../nlpia2/src/nlpia2/ch07/cnn-stencil-right-ones-right-order-pos-one-hot-vec-sequence.md
