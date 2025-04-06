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
hist -o -p -f char_rnn_name_nationality_nonrandom_sample.hist.md
hist -o -p
df
>>> fraction_unique = {}
>>> for i, g in df.groupby('category'):
...     fraction_unique[i] = g['name'].nunique() / len(g)
>>> pd.Series(fraction_unique).sort_values()
from torch.nn import RNN
htop
!htop
results = train()
from collections import Counter
>>> confusion = {c: Counter() for c in CATEGORIES}
>>> counts = {}
>>> for i, g in df.groupby('name'):
...      counts = Counter(g['category']) 
...      most_popular = sorted([(x[1], x[0]) for x in zip(counts.items())])[-1][1]
...      confusion[most_popular] += counts
Counter(dict(a=2, b=3))
from collections import Counter
>>> confusion = {c: Counter() for c in CATEGORIES}
>>> counts = {}
>>> for i, g in df.groupby('name'):
...      counts = Counter(g['category']) 
...      most_popular = sorted([(x[1], x[0]) for x in counts.items()])[-1][1]
...      confusion[most_popular] += counts
pd.DataFrame(confusion)
confusion = pd.DataFrame(confusion)
confusion /= confusion.sum(axis=1)
confusion
confusion.fillna(0, inplace=True)
confusion
confusion.round(2)
hist -o -p -f char_rnn_name_nationality_confusion.hist.md
confusion = confusion[confusion.index]
confusion
confusion.round(2)
confusion.plot?
import seaborn
cax = plt.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
fig = figure()
fig = plt.figure()
cax = plt.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
ax = fig.add_subplot(111)
cax = plt.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + confusion.columns, rotation=90)
ax.set_yticklabels([''] + confusion.columns)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
hist
seaborn.set_style()
seaborn.set_theme()
hist -o -p -f char_rnn_name_nationality_dataset_confusion.hist.md
hist -f char_rnn_name_nationality_dataset_confusion.hist.py
