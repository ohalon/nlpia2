dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000)):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
import pandas as pd
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000)):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz', 'b') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000)):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
import gzip
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz', 'rb') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000)):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz', 'rb', encoding='latin') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000)):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz', 'rb') as fin:
    for i, df in enumerate(pd.read_csv(fin, chunksize=1000, encoding='latin')):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        print(i, line, len(line.split(',')))
        dfs.append(line)
        if i > 10:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        line = line.decode('latin')
        print(i, line, len(line.split(',')))
        dfs.append(line)
        if i > 10:
            break
dfs = []
with open('../grounder/data/NELL.08m.1115.esv.csv.gz', 'rb') as fin:
    for i, df in enumerate(pd.read_csv(fin, sep='\t', chunksize=1000, encoding='latin')):
        print(i, df.head())
        dfs.append(df)
        if i > 10:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        print(i, len(line.split('\t')))
        dfs.append(i, line)
        if i > 1000:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        line = line.decode('latin')
        print(i, len(line, split('\t')))
        dfs.append(i, line)
        if i > 1000:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        line = line.decode('latin')
        print(i, len(line.split('\t')))
        dfs.append(i, line)
        if i > 1000:
            break
dfs = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        line = line.decode('latin')
        print(i, len(line.split('\t')))
        dfs.append([i, line])
        if i > 1000:
            break
lines = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in enumerate(fin):
        line = line.decode('latin')
        print(i, len(line.split('\t')))
        dfs.append(line.split('\t'))
        if i > 100000:
            break
lines = []
lineno = 0
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in tqdm(enumerate(fin)):
        lineno = i
        # line = line.decode('latin')
        # print(i, len(line.split('\t')))
        # dfs.append(line.split('\t'))
from tqdm import tqdm
lines = []
lineno = 0
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in tqdm(enumerate(fin)):
        lineno = i
        # line = line.decode('latin')
        # print(i, len(line.split('\t')))
        # dfs.append(line.split('\t'))
lineno
lines = []
with gzip.open('../grounder/data/NELL.08m.1115.esv.csv.gz') as fin:
    for i, line in tqdm(enumerate(fin), total=lineno):
        line = line.decode('latin')
        # print(i, len(line.split('\t')))
        lines.append(line.split('\t'))
df = pd.DataFrame(lines)
df.shape
df.head()
ls /home/hobs/Dropbox/Public/videos/
pwd
hist -o -p -f 2023-05-29-nell-candidate-beliefs.hist.ipy
hist -f 2023-05-29-nell-candidate-beliefs.hist.py
