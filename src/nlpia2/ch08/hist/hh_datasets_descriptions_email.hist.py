import huggingface_hub
import pandas as pd
import datasets as ds

hh = huggingface_hub
infos = hh.list_datasets()

df = pd.DataFrame()
df['description'] = [info.description for info in infos]
df['name'] = [info.id for info in infos]
keys = 'tags gated paperswithcode_id downloads author citation'.split()
for k in keys:
    print(k)
    df[k] = [getattr(i, k, None) for i in infos]
is_email = pd.Series(['email' in d.lower().replace('-', '') if d else False for d in df['description']])

"""
>>> df[is_email.values].iloc[0]['description']
>>> df[is_email.values].iloc[1]['description']
"""

enron_emails = ds.load_dataset(path=df.iloc[3]['name'])
print(enron_emails['test'].keys())
print(enron_emails['test']['subject_line'][0])
print(enron_emails['test']['email_body'][0])
