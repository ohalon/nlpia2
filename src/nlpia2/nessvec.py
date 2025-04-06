""" The nessvec package, imported as a module within nlpia2

### FIXME (2nd edition):
#### `nessvec.load_vecs_df(uri)`
Should work with bz2, gz, csv, .txt absolute file paths, file names (in HOME_DATA_DIR) & remote URLs:
>>> glove = load_vecs_df(HOME_DATA_DIR / 'glove.6B.50d.txt')
>>> glove = load_vecs_df(HOME_DATA_DIR / 'glove.6B.50d.txt.bz2')
>>> url = 'https://gitlab.com/tangibleai/nlpia2/-'
>>> url += '/raw/main/src/nlpia2/data/glove.6B.50d.txt.bz2'
>>> glove = load_vecs_df(url)

#### `nessvec.download`:
Download a URL to HOME_DATA_DIR and return local path. Should work like wget:
!wget https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2/data/glove.6B.50d.txt.bz2

#### `nessvec.decompress`:
Decompress a binary zip/gz/bz2 file and convert bytes to strings:
>>> import bz2
>>> glovebytes = open('glove.6B.50d.txt.bz2', 'rb').read()
>>> glovetext = bz2.decompress(glovebytes).decode()
>>> with open('glove.6B.50d.txt', 'w') as fout:
...     fout.write(glovetext)
171337876
"""

from nessvec import *  # noqa