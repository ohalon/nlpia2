# Dota 2 Hero character sheets

# Dota2 fandom have created some nice tabular data describing all the Dota 2 heros.


import pandas as pd
dfs = pd.read_html('https://dota2.fandom.com/wiki/Strength')
dfs[0]
#     Hero             Hero.1  ...  Base Intelligence  Intelligence Growth
# 0    NaN            Abaddon  ...                 18                  2.0
# 1    NaN          Alchemist  ...                 25                  1.8
# 2    NaN                Axe  ...                 18                  1.6
# 3    NaN        Beastmaster  ...                 16                  1.9
# ...
# 38   NaN               Tusk  ...                 18                  1.7
# 39   NaN          Underlord  ...                 17                  2.3
# 40   NaN            Undying  ...                 27                  2.8
# 41   NaN        Wraith King  ...                 18                  1.6

# [42 rows x 8 columns]


# Get tables for the other 2 hero categories.


categories = 'Strength Agility Intelligence'.split()
url = 'https://dota2.fandom.com/wiki'
df = pd.concat([pd.read_html(f'{url}/{c}')[0].fillna(c) for c in categories])


# What about some NL text describing each hero:


df = pd.read_html('https://dota2.fandom.com/wiki/Heroes/Introductions')[1]
attr = pd.concat([pd.read_html(f'{url}/{c}')[0].fillna(c) for c in categories])
attrs = attr
dfs = pd.read_html(f'{url}/Heroes/Introductions')[1:]
intros = pd.concat(dfs).set_index('Hero')
intros
#                                                      Description
# Hero
# Abaddon        Unlike the other scions of his house, Abaddon ...
# Alchemist      Having broken out of prison with his ogre acco...
# Axe            One by one the soldiers of the Red Mist fell, ...
# Beastmaster    Raised in the royal menagerie of Slom, Karroch...
# Brewmaster     For nine days Mangix fought the master of his ...
# ...                                                          ...
# Warlock        In the endless pursuit of rare texts, Demnok L...
# Windranger     Born in the midst of a ruinous tempest, Lyrale...
# Winter Wyvern  Frustrated with writer's block, Auroth, the El...
# Witch Doctor   A bizarre figure shambles across the land, sea...
# Zeus           Fond of mortal women, but even fonder of immor...

# [123 rows x 1 columns]


# Consolidated table of numerical attributes for each hero


attrs2 = pd.read_html(f'{url}/Table_of_hero_attributes')[0]
attrs2
#                    HERO   A  STR  STR+  STR30  ...  VS-N   TR  COL  HP/S  L
# 0               Abaddon NaN   22   2.6   97.4  ...   800  0.6   24  3.20  2
# 1             Alchemist NaN   25   2.9  109.1  ...   800  0.6   24  2.75  2
# 2    Ancient Apparition NaN   20   1.9   75.1  ...   800  0.6   24  2.25  0
# 3             Anti-Mage NaN   21   1.6   67.4  ...   800  0.6   24  2.35  2
# 4            Arc Warden NaN   22   2.6   97.4  ...   800  0.7   24  2.45  2
# ..                  ...  ..  ...   ...    ...  ...   ...  ...  ...   ... ..
# 118          Windranger NaN   18   3.0  105.0  ...   800  0.8   24  2.05  2
# 119       Winter Wyvern NaN   24   2.6   99.4  ...   800  0.6   24  2.65  2
# 120        Witch Doctor NaN   18   2.3   84.7  ...   800  0.6   24  2.05  2
# 121         Wraith King NaN   22   3.0  109.0  ...   800  0.6   24  2.45  2
# 122                Zeus NaN   19   2.1   79.9  ...   800  0.6   24  2.15  2

# [123 rows x 29 columns]


# Only thing missing now is the release date/year/version info:


releases = pd.read_html(f'{url}/Heroes_by_release')
releases = releases[0].set_index('Hero')


# Clean up the column names and indices:


attrs.columns = ['Dominant Attribute', 'Hero'] + list(attrs.columns[2:])
attrs = attrs.set_index('Hero')
attrs2 = attrs2.set_index('HERO')


# Make sure we have all the heroes' names in all the table indices:


names2 = set(attrs2.index)
names = set(attrs.index)
names - names2
# set()
names2 - names
# set()
names == names2
# True


# Now the `releases` DataFrame index:


releases.index = [i[0] for i in releases.index]
releases
#                     Dota 2         DotA Allstars Version
#                    Release  Status       Release Version
# Anti-Mage       2010-11-01   Alpha    2004-03-01    2.60
# Axe             2010-11-01   Alpha    2005-03-01    6.00
# Crystal Maiden  2010-11-01   Alpha    2004-02-03    0.60
# Dazzle          2010-11-01   Alpha    2005-03-01    6.00
# Drow Ranger     2010-11-01   Alpha    2004-02-03    0.60
# ...                    ...     ...           ...     ...
# Void Spirit     2019-11-26  Reborn             -    7.23
# Hoodwink        2020-12-17  Reborn             -    7.28
# Dawnbreaker     2021-04-09  Reborn             -    7.29
# Marci           2021-10-28  Reborn             -   7.30e
# Primal Beast    2022-02-23  Reborn             -    7.31

# [123 rows x 4 columns]
releases.columns
# MultiIndex([(       'Dota 2', 'Release'),
#             (       'Dota 2',  'Status'),
#             ('DotA Allstars', 'Release'),
#             (      'Version', 'Version')],
#            )
releases.columns.get_level_values(1)
# Index(['Release', 'Status', 'Release', 'Version'], dtype='object')
releases.columns = releases.columns.get_level_values(1)
# releases
#                    Release  Status     Release Version
# Anti-Mage       2010-11-01   Alpha  2004-03-01    2.60
# Axe             2010-11-01   Alpha  2005-03-01    6.00
# Crystal Maiden  2010-11-01   Alpha  2004-02-03    0.60
# Dazzle          2010-11-01   Alpha  2005-03-01    6.00
# Drow Ranger     2010-11-01   Alpha  2004-02-03    0.60
# ...                    ...     ...         ...     ...
# Void Spirit     2019-11-26  Reborn           -    7.23
# Hoodwink        2020-12-17  Reborn           -    7.28
# Dawnbreaker     2021-04-09  Reborn           -    7.29
# Marci           2021-10-28  Reborn           -   7.30e
# Primal Beast    2022-02-23  Reborn           -    7.31

# [123 rows x 4 columns]


# Confirm releases.index contains all the hero names:


names3 = set(releases.index)
names3
# {'Abaddon',
#  'Alchemist',
#  'Ancient Apparition',
#  'Anti-Mage',
# ...
#  'Witch Doctor',
#  'Wraith King',
#  'Zeus'}
names3 - names
# set()
names - names3
# set()
names == names3
# True


# Finally the intros:


intros.head()
#                                                    Description
# Hero
# Abaddon      Unlike the other scions of his house, Abaddon ...
# Alchemist    Having broken out of prison with his ogre acco...
# Axe          One by one the soldiers of the Red Mist fell, ...
# Beastmaster  Raised in the royal menagerie of Slom, Karroch...
# Brewmaster   For nine days Mangix fought the master of his ...
names4 = set(intros.index)
names == names4
# True


# Combine all the things:


df = pd.concat([attrs, attrs2, intros, releases], axis=1)
from nlpia2.constants import DATA_DIR  # noqa
from pathlib import Path  # noqa
df.to_csv(Path(DATA_DIR) / "dota2-heroes.csv")
df.head()
#             Dominant Attribute  Base Strength  Strength Growth  Base Agility  ...     Release  Status     Release  Version
# Abaddon               Strength             22              2.6            23  ...  2013-07-12    Gold  2005-11-04     6.20
# Alchemist             Strength             25              2.9            22  ...  2011-11-18    Beta  2006-08-25     6.36
# Axe                   Strength             25              3.4            20  ...  2010-11-01   Alpha  2005-03-01     6.00
# Beastmaster           Strength             23              2.9            18  ...  2011-05-20    Beta  2006-04-11     6.30
# Brewmaster            Strength             23              3.7            19  ...  2012-04-19    Beta  2004-04-30     5.10

# [5 rows x 40 columns]


# Magic to create the history logs used to build these docs:


# %hist -o -p -f /home/hobs/code/tangibleai/nlpia2/examples/ch06_dota2_wiki_heroes.hist.ipy.md
# %hist -o -p -f /home/hobs/code/tangibleai/nlpia2/ch06_dota2_wiki_heroes.hist.ipy.md
