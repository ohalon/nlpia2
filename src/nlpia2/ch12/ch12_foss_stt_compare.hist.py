import pandas as pd
pd.read_html('https://deepgram.com/learn/benchmarking-top-open-source-speech-models#evaluating-accuracy')
dfs = _
for df in dfs
df = dfs[0]
df.set_index('0')
df.columns
df.set_index(df.columns[0])
df = df.set_index(df.columns[0])

for n, dfi in zip(names, dfs):
    if 'Domain' in dfi.columns:
        dfi = dfi.set_index['Domain']
        for c in dfi.columns:
        df[n + '_' + c] = dfi

for n, dfi in zip(names, dfs):
    if 'Domain' in dfi.columns:
        dfi = dfi.set_index['Domain']
        for c in dfi.columns:
            df[n + '_' + c] = dfi
names = [str(i) for i in range(1, len(dfs))]

for n, dfi in zip(names, dfs[1:]):
    if 'Domain' in dfi.columns:
        dfi = dfi.set_index['Domain']
        for c in dfi.columns:
            df[n + '_' + c] = dfi
df
dfs[1]

for n, dfi in zip(names, dfs[1:]):
    if 'Dataset' in dfi.columns:
        dfi = dfi.set_index['Dataset']
        for c in dfi.columns:
            df[n + '_' + c] = dfi

for n, dfi in zip(names, dfs[1:]):
    if 'Dataset' in dfi.columns:
        dfi = dfi.set_index('Dataset')
        for c in dfi.columns:
            df[n + '_' + c] = dfi

for n, dfi in zip(names, dfs[1:]):
    print(n, dfi.index.name, list(dfi.columns))
    if 'Dataset' in dfi.columns:
        dfi = dfi.set_index('Dataset')
        for c in dfi.columns:
            df[n + '_' + c] = dfi
df.columns

for n, dfi in zip(names, dfs[1:]):
    print(n, dfi.index.name, list(dfi.columns))
    if 'Dataset' in dfi.columns:
        dfi = dfi.set_index('Dataset')
        for c in dfi.columns:
            df[n + '_' + c] = dfi[c]
df
df.T
df.index
dfi.colu,ns
dfi
df
df.T
pd.options.display.max_colwidth = 7
df.T
dfi['Dataset']
df.T

for n, dfi in zip(names, dfs[1:]):
    print(dfi.index.name)
    if 'Dataset' in dfi.columns:
        dfi = dfi.set_index('Dataset')
        print(n, dfi.index.name, list(dfi.columns))    
        for c in dfi.columns:
            print('   ' + c)
            df[n + '_' + c] = dfi[c]
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1)
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1, ignore_index=True)
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1)
pd.options.display.max_colwidth = 5
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1)
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1).T
pd.options.display.max_colwidth = 10
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:]], axis=1).T
pd.concat([d[list(d.columns)[1:]] for d in dfs[1:-3]], axis=1).T
df2 = pd.concat([d[list(d.columns)[1:]] for d in dfs[1:-3]], axis=1)
df2.index = df.index
df2
df2.T
df2.index = 'Converstion Phone Meeting Video Finance'.split()
df2
df2.T
df.index = 'Converstion Phone Meeting Video Finance'.split()
pd.concat([df, df2], axis=1)
df
dfs[0]
df0 = dfs[0]
df0.index = 'Converstion Phone Meeting Video Finance'.split()
df0
df0.drop('Domain')
df0.drop('Domain', axis=1)
df0 = df0.drop('Domain', axis=1)
df0
df2['Description'] = df0[df0.columns[0]]
df2
df2.T
df2.index = 'AI Phone Meeting Video Finance'.split()
df2.T
ls -hal
hist -o -p -f ~/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare.hist.ipy
hist -f ~/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare.hist.py
dfs[4]
dfs[5]
dfs[6]
dfs[7]
dfs[5]
[d['Whisper'].iloc[0] for d in dfs]
[d['Whisper'].iloc[0] for d in dfs if 'Whisper' in d.columns]
dfs[5] = dfs[5].set_index('Dataset')
df_ = dfs[5]
df_
df_.to_markdown()
df_.to_markdown?
with open('~/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare_table5.md') as fout:
    fout.write(df_.to_markdown())
with open('/home/hobs/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare_table5.md') as fout:
    fout.write(df_.to_markdown())
with open('/home/hobs/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare_table5.md', 'r+') as fout:
    fout.write(df_.to_markdown())
with open('/home/hobs/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare_table5.md', 'w+') as fout:
    fout.write(df_.to_markdown())
more /home/hobs/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare_table5.md
df_.index = 'AI Phone Meeting Video Finance'.split()
print(df_.to_markdown())
df_.index.name='Domain'
print(df_.to_markdown())
print(df_.round(0).to_markdown())
print(df_.T.round(0).to_markdown())
df_['Ave'] = df_.mean(axis=1)
df_
df_['Ave']
df_.drop('Ave')
df_ = df_.drop('Ave', axis=1)
df_
df_['Mean'] = df_.T.mean()
df_
print(df_.T.round(0).to_markdown())
df_ = df_.drop('Mean', axis=1)
df = df_.T
df.mean(axis=1)
df['Mean'] = df.mean(axis=1)
df
df.to_markdown()
print(df.to_markdown())
print(df.round().to_markdown())
df
print(df.round().to_markdown())
df = df.round()
df.apply(lambda x: f'{x}%')
df
df[c] = df[c].apply(lambda x: f'{x}%')
for c in df.columns: df[c] = df[c].apply(lambda x: f'{x}%')
df
for c in df.columns: df[c] = df[c].apply(lambda x: x[:-3]+'%')
df
print(df.to_markdown())
hist -f ~/code/tangibleai/community/nlpia2/src/nlpia2/ch12/ch12_foss_stt_compare.hist.py
