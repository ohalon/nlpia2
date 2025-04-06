from nlpia2.string_normalizers import *
%run rnn_from_scratch_name_nationality.py
%run rnn_from_scratch_name_nationality.py
import huggingface_hub
import pandas as pd

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
df
is_email

import datasets as ds

ds.load_dataset(path=df.iloc[3]['name'])
enron_emails['test']['email_body'][0]
enron_emails = ds.load_dataset(path=df.iloc[3]['name'])
enron_emails['test']['subject_line'][0]
enron_emails['test'].data.column_names
print(enron_emails['test']['email_body'][0])
enron_emails = ds.load_dataset(path=df.iloc[3]['tags'])
df.iloc[3]['tags']
df.iloc[2]['tags']
df.iloc[1]['tags']
df.iloc[0]['tags']
df.iloc[3]['name']
df.iloc[3]['tags']
alltags = [tags for tags in df.iloc[3]['tags']]
task_categories = [[t for t in tags if t.startswith('task_categories')] for tags in alltags]
possible_task_cagegories = set(sum(task_categories))
task_categories
alltags
alltags = [tags for tags in df['tags'].values]
task_categories = [[t for t in tags if t.startswith('task_categories')] for tags in df['tags'].values]
possible_task_cagegories = set(sum(task_categories))
task_categories
possible_task_cagegories = sum(task_categories)
possible_task_cagegories = set(list(chain(*task_categories)))
from itertools import chain
possible_task_cagegories = set(list(chain(*task_categories)))
possible_task_cagegories
task_ids = [[t for t in tags if t.startswith('task_id')] for tags in df['tags'].values]
len(possible_task_cagegories)
task_categories = [[t for t in tags if t.startswith('task_categor')] for tags in df['tags'].values]
possible_task_cagegories = set(list(chain(*task_categories)))
len(possible_task_cagegories)
possible_task_ids = set(list(chain(*task_ids)))
possible_task_ids
len(possible_task_ids)
