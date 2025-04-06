df = pd.read_csv('https://dota2.fandom.com/wiki/Strength')
dfs = pd.read_html('https://dota2.fandom.com/wiki/Strength')
import pandas as pd
dfs = pd.read_html('https://dota2.fandom.com/wiki/Strength')
dfs[0]
dfs[1]
categories = 'Strength Agility Intelligence'.split()
pd.concat([pd.read_html(f'https://dota2.fandom.com/wiki/{c}')[0].fillna(c) for c in categories], axis=0)
pd.concat([pd.read_html(f'https://dota2.fandom.com/wiki/{c}')[0].fillna(c) for c in categories], axis=1)
pd.concat([pd.read_html(f'https://dota2.fandom.com/wiki/{c}')[0].fillna(c) for c in categories])
df = _
url = 'https://dota2.fandom.com/wiki'
df = pd.concat([pd.read_html(f'{url}/{c}')[0].fillna(c) for c in categories])
df.sum()
pd.read_html('https://dota2.fandom.com/wiki/Heroes/Introductions')
pd.read_html('https://dota2.fandom.com/wiki/Heroes/Introductions')[0]
pd.read_html('https://dota2.fandom.com/wiki/Heroes/Introductions')[1]
df =pd.read_html('https://dota2.fandom.com/wiki/Heroes/Introductions')[1]
attr = pd.concat([pd.read_html(f'{url}/{c}')[0].fillna(c) for c in categories])
attrs = attr
dfs = pd.read_html(f'{url}/Heroes/Introductions')[1:]
dfs
pd.concat(dfs)
intros = pd.concat(dfs).set_index('Hero')
intros
intros['Heros']
intros['Hero']
attrs2 = pd.read_html(f'{url}/Table_of_hero_attributes')
attrs2[0]
attrs2[1]
attrs2 = pd.read_html(f'{url}/Table_of_hero_attributes')[0]
releases = pd.read_html(f'{url}/Heroes_by_release')
realeases[0]
releases[0]
releases = releases.set_index('Hero')
len(releases)
releases = releases[0].set_index('Hero')
who
attrs.set_index('Hero.1')
attrs.columns = ['Hero', 'Domnant Attribute'] + attrs.columns[2:]
attrs.columns = ['Hero', 'Domnant Attribute'] + list(attrs.columns[2:])
attrs
attrs.columns = ['Domnant Attribute', 'Hero'] + list(attrs.columns[2:])
attrs
attrs.columns = ['Dominant Attribute', 'Hero'] + list(attrs.columns[2:])
attrs = attrs.set_index('Hero')
who
attrs2
attrs2 = attrs2.set_index('HERO')
names2 = set(attrs2.index)
names = set(attrs.index)
names - names2
names2 - names
names ==  names2
releases
releases.index = [releases.index[0]]
releases.index = [i[0] for i in releases.index]
releases
releases.columns
releases.columns.get_level_values(1)
releases.columns = releases.columns.get_level_values(1)
releases
names3 = set(releases.index)
names3
names3 - names
names - names3
names == names3
who
intros.head()
names == names4
names4 = set(intros.index)
names == names4
pd.concat([attrs2, attrs, intros, releases])
len(names4)
len(intros)
pd.concat([intros, releases])
pd.concat([attrs, intros])
pd.concat([attrs, attrs2, intros])
pd.concat([attrs, attrs2, intros], axis=1)
pd.concat([attrs, attrs2, intros, releases], axis=1)
df = pd.concat([attrs, attrs2, intros, releases], axis=1)
df.to_csv(DATA_DIR / "dota2-heroes.csv")
from nlpia2.constants import DATA_DIR
from qary.constants import DATA_DIR
df.to_csv(DATA_DIR / "dota2-heroes.csv")
df.to_csv(DATA_DIR + "/dota2-heroes.csv")
df.head()
%hist -o -p -f /home/hobs/code/tangibleai/nlpia2/examples/ch06_dota2_wiki_heroes.hist.ipy.md
%hist -o -p -f /home/hobs/code/tangibleai/nlpia2/ch06_dota2_wiki_heroes.hist.ipy.md
%hist -f /home/hobs/code/tangibleai/nlpia2/ch06_dota2_wiki_heroes.hist.py
