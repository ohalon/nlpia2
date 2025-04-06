>>> import nessvec
>>> import nlpia2
>>> from nlpia2.constants import DATA_DIR, HOME_DATA_DIR
>>> import pandas as pd
>>> filepath = HOME_DATA_DIR / 'disaster-tweets.csv'
>>> pd.read_csv(filepath, index_col=0)
      keyword location                                               text  target
id                                                                               
1         NaN      NaN  Our Deeds are the Reason of this #earthquake M...       1
4         NaN      NaN             Forest fire near La Ronge Sask. Canada       1
5         NaN      NaN  All residents asked to 'shelter in place' are ...       1
6         NaN      NaN  13,000 people receive #wildfires evacuation or...       1
7         NaN      NaN  Just got sent this photo from Ruby #Alaska as ...       1
...       ...      ...                                                ...     ...
10869     NaN      NaN  Two giant cranes holding a bridge collapse int...       1
10870     NaN      NaN  @aria_ahrary @TheTawniest The out of control w...       1
10871     NaN      NaN  M1.94 [01:04 UTC]?5km S of Volcano Hawaii. htt...       1
10872     NaN      NaN  Police investigating after an e-bike collided ...       1
10873     NaN      NaN  The Latest: More Homes Razed by Northern Calif...       1

[7613 rows x 4 columns]
>>> df = pd.read_csv(filepath, index_col=0)
>>> df = pd.read_yaml(HOME_DATA_DIR / 'quotes.yml')
>>> pd.json_normalize(safe_load(fin))
>>> fin = open(HOME_DATA_DIR / 'quotes.yml')
>>> import yaml
>>> from yaml import safe_load
>>> df = pd.json_normalize(safe_load(fin))
>>> df = pd.json_normalize(safe_load(fin))
>>> fin = open(HOME_DATA_DIR / 'quotes.yml')
>>> df = pd.json_normalize(safe_load(fin))
>>> df = pd.json_normalize(safe_load(fin))
>>> fin = open(HOME_DATA_DIR / 'quotes.yml')
>>> df = pd.json_normalize(safe_load(fin))
>>> df = pd.json_normalize(safe_load(fin))
>>> fin = open(HOME_DATA_DIR / 'quotes.yml')
>>> df = pd.json_normalize(safe_load(fin))
>>> with open(HOME_DATA_DIR / 'quotes.yml') as fin:
...     df = pd.json_normalize(safe_load(fin))
...
>>> with open(HOME_DATA_DIR / 'quotes.yml') as fin:
...     df = pd.json_normalize(safe_load(fin))
...
>>> with open(HOME_DATA_DIR / 'quotes.yml') as fin:
...     df = pd.json_normalize(safe_load(fin))
...
>>> with open(HOME_DATA_DIR / 'quotes.yml') as fin:
...     df = pd.json_normalize(safe_load(fin))
...
>>> with open(HOME_DATA_DIR / 'quotes.yml') as fin:
...     df = pd.json_normalize(safe_load(fin))
...
>>> df.shape
(240, 30)
>>> df.head()
                                                text  ... references
0  If you disagree with somebody, you want to be ...  ...        NaN
1  The man who can hold forth on every matter und...  ...        NaN
2  I have yet to find a more efficient and reliab...  ...        NaN
3  If you can't imagine how anyone could hold the...  ...        NaN
4  When you think you can nail someone with your ...  ...        NaN

[5 rows x 30 columns]
>>> df['target'] = -1
>>> git status
>>> meld tangibleai/nlpia2/src/nlpia2/data/quotes.yml ~/.nlpia2-data/quotes.yml
>>> posts = pd.read_csv(filepath, index_col=0)
>>> quotes = df[['text', 'target']]
>>> pd.concat([posts, quotes])
    keyword location                                               text  target
1       NaN      NaN  Our Deeds are the Reason of this #earthquake M...       1
4       NaN      NaN             Forest fire near La Ronge Sask. Canada       1
5       NaN      NaN  All residents asked to 'shelter in place' are ...       1
6       NaN      NaN  13,000 people receive #wildfires evacuation or...       1
7       NaN      NaN  Just got sent this photo from Ruby #Alaska as ...       1
..      ...      ...                                                ...     ...
235     NaN      NaN  Individuality is deeply imbued and us from the...      -1
236     NaN      NaN  Similar considerations arise with regard to re...      -1
237     NaN      NaN  And in its broadest sense, neural Darwinism im...      -1
238     NaN      NaN  In the 1980's, Edelman's theory was so novel t...      -1
239     NaN      NaN  Over a lifetime, I have written millions of wo...      -1

[7853 rows x 4 columns]
>>> df = pd.concat([posts, quotes])
>>> df.reset_index()
      index keyword location                                               text  target
0         1     NaN      NaN  Our Deeds are the Reason of this #earthquake M...       1
1         4     NaN      NaN             Forest fire near La Ronge Sask. Canada       1
2         5     NaN      NaN  All residents asked to 'shelter in place' are ...       1
3         6     NaN      NaN  13,000 people receive #wildfires evacuation or...       1
4         7     NaN      NaN  Just got sent this photo from Ruby #Alaska as ...       1
...     ...     ...      ...                                                ...     ...
7848    235     NaN      NaN  Individuality is deeply imbued and us from the...      -1
7849    236     NaN      NaN  Similar considerations arise with regard to re...      -1
7850    237     NaN      NaN  And in its broadest sense, neural Darwinism im...      -1
7851    238     NaN      NaN  In the 1980's, Edelman's theory was so novel t...      -1
7852    239     NaN      NaN  Over a lifetime, I have written millions of wo...      -1

[7853 rows x 5 columns]
>>> df = pd.concat([posts, quotes]).reset_index()[['text', 'target', 'keyword']]
>>> df
                                                   text  target keyword
0     Our Deeds are the Reason of this #earthquake M...       1     NaN
1                Forest fire near La Ronge Sask. Canada       1     NaN
2     All residents asked to 'shelter in place' are ...       1     NaN
3     13,000 people receive #wildfires evacuation or...       1     NaN
4     Just got sent this photo from Ruby #Alaska as ...       1     NaN
...                                                 ...     ...     ...
7848  Individuality is deeply imbued and us from the...      -1     NaN
7849  Similar considerations arise with regard to re...      -1     NaN
7850  And in its broadest sense, neural Darwinism im...      -1     NaN
7851  In the 1980's, Edelman's theory was so novel t...      -1     NaN
7852  Over a lifetime, I have written millions of wo...      -1     NaN

[7853 rows x 3 columns]
>>> quotes.columns
Index(['text', 'target'], dtype='object')
>>> df.to_csv(DATA_DIR / 'deep-posts.csv', index=None)
>>> posts.shape
(7613, 4)
>>> df = pd.read_csv(DATA_DIR / 'deep-posts.csv')
>>> pwd
'/home/hobs/code/tangibleai/nlpia2'
>>> hist -o -p -f src/nlpia2/scripts/deep-posts.hist.md
