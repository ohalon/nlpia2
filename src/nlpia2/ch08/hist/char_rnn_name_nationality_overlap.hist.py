from char_rnn_from_scratch_refactored import *
df = load_dataset()
filename = 'char_rnn_from_scratch_refactored-1_311-17min_28sec'
df
prefix = df['name'].str[:3]
prefix
prefix_nationality = zip(prefix, df['category'])
from collections import Counter
Counter(prefix_nationality.values)
prefix_nationality = list(zip(prefix, df['category']))
Counter(prefix_nationality)
pd.Series(Counter(prefix_nationality))
df.groupby('name')
for g in df.groupby('name'):
    print(g['nationality'].nunique())
g
for i, g in df.groupby('name'):
    n = g['nationality'].nunique()
    if n > 1:
        print(f"{i}: {n}")
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        print(f"{i}: {n}")
overlap = {}
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        print(f"{i}: {n}")
        overlap[i] = n
overlap = {}
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        overlap[i] = n
pd.Series(overlap)
pd.Series(overlap).sort_values()
pd.Series(overlap).sort_values(ascending=False)
overlap = {}
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        overlap[i] = list(g['category'].unique())
pd.Series(overlap).sort_values(ascending=False)
overlap = {}
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        overlap[i] = [n, list(g['category'].unique())]
overlap = {}
for i, g in df.groupby('name'):
    n = g['category'].nunique()
    if n > 1:
        overlap[i] = {'nunique': n, 'unique': list(g['category'].unique())}
pd.DataFrame(overlap)
overlap = pd.DataFrame(overlap).T
overlap.sort_values('nunique')
overlap.sort_values('nunique', ascending=False)
hist
hist -o -p -f char_rnn_name_nationality_overlap.hist.md
hist -f char_rnn_name_nationality_overlap.hist.py
