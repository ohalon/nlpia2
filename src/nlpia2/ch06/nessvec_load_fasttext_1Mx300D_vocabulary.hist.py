import nessvec
from nessvec.files import load_fasttext
df = load_fasttext()
df['token'] = df.index.values
df['token'].str.contains(' ').sum()
df['token'].str.contains('_').sum()
df['token'].str.contains('-').sum()
df['token'].str.contains(',')]
df[df['token'].str.contains('-')].head()
df[df['token'].str.contains('-')].sample(100)['token']
df[df['token'].str.contains('-')].sample(10)['token']
df[df['token'].str.contains('.')].sample(10)['token']
df[df['token'].str.contains(r'[.]')].sample(10)['token']
df[df['token'].str.contains(r'\[')].sample(10)['token']
df[df['token'].str.contains(r'[\[]')].sample(10)['token']
df[df['token'].str.contains(r'[[]')].sample(10)['token']
df[df['token'].str.contains(r'[^]')].sample(10)['token']
df[df['token'].str.contains(r'[\^]')].sample(10)['token']
df[df['token'].str.contains(r'[\^]')].sample()['token']
df[df['token'].str.contains(r'[\~]')].sample()['token']
df[df['token'].str.contains(r'[\~]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[\!]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[\_]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[ ]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[|]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[,]')].sample(3, replace=True)['token']
df[df['token'].str.contains(r'[@]')].sample(3, replace=True)['token']
hist -o -p -f 'src/nlpia2/ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md'
ls
pwd
hist -o -p -f 'ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md'
hist -o -p -f ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md
hist -f ch06/nessvec_load_fasttext_1Mx300D_vocabulary.py
hist -f ch06/nessvec_load_fasttext_1Mx300D_vocabulary.hist.py
