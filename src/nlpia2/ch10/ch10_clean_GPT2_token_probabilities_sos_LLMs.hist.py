from nlpia2.constants import SRC_DATA_DIR
import pandas as pd
df = pd.read_csv(SRC_DATA_DIR / 'ch10_GPT2_token_probabilities_sos_LLMs.csv')
df
df = pd.read_csv(SRC_DATA_DIR / 'ch10_GPT2_token_probabilities_sos_LLMs.csv', index_col=0)
df
d['are'] = df['LLMs'].values
d = {}
d['are'] = df['LLMs'].values
d['are'] = df['LLMs'].str.split()
d
d['are'] = df['LLMs'].str.split().apply(l: [l[0], l[-2]])
d['are'] = df['LLMs'].str.split().apply(lambda l: [l[0], l[-2]])
d
d['are'] = df['LLMs'].str.split().apply(lambda l: [l[0], l[-2][1:]])
df2 = pd.DataFrame()
    d[c] = df[c].str.split().apply(lambda l: [l[0], l[-2][1:]])
df.columns = list(df.columns[1:]) + ['the']
df
df
df2 = pd.DataFrame()
df2['LLMs'] = [''] * 6
for c in df.columns:
    df2[c] = df[c].str.split().apply(lambda l: ' '.join([l[0], l[-2][1:].lstrip('0')]))
df.fillna('? 0.0% _')
for c in df.columns:
    df2[c] = df[c].str.split().apply(lambda l: ' '.join([l[0], l[-2][1:].lstrip('0')]))
df
df.fillna('? 0.0% _', inplace=True)
for c in df.columns:
    df2[c] = df[c].str.split().apply(lambda l: ' '.join([l[0], l[-2][1:].lstrip('0')]))
df2
df2['most'] = ['most']+ [''] * 5
df2
del df2['most']
df2
df2.columns = [c.strip() for c in df2.columns]
df2['most'] = ['most'] + [''] * 5
df2['LLMs'] = ['LLMs'] + [''] * 5
df2
df2.to_csv(SRC_DATA_DIR / 'ch10_GPT2_token_probabilities_sos_LLMs.cleaned.csv', index=False)
pwd
hist -o -p -f ../nlpia2/src/nlpia2/ch10/ch10_clean_GPT2_token_probabilities_sos_LLMs.hist.ipy
hist -f ../nlpia2/src/nlpia2/ch10/ch10_clean_GPT2_token_probabilities_sos_LLMs.hist.py
