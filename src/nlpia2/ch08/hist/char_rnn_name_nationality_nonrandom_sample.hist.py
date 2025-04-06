ls -hal
df = load_dataset()
from collections import Counter
from char_rnn_from_scratch_refactored import *
df = load_dataset()
groups = df.groupby('category')
for i, g in groups:
    print(i, g['text'].nunique() / len(g))
g.columns
for i, g in groups:
    print(i, g['name'].nunique() / len(g))
len(groups['Arabic'])
groups
g = df[df['category'] == 'Arabic']
len(g)
g
g['name'].sort_values()
hist -o -p
hist -f char_rnn_name_nationality_nonrandom_sample.hist.py
