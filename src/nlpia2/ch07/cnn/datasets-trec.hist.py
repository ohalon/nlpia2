import datasets
datasets.search('TREC')
from datasets import *
who
dir(search)
vars(search)
dir(search)
who
dataset_dict
datasets?
help(datasets)
tasks
dir(tasks)
dir(tasks.TextClassification)
help(tasks.TextClassification)
trec = load_dataset('trec')
trecqa = load_dataset('trec-qa')
trecqa = load_dataset('trec-qc')
trecqa = load_dataset('SetFit/TREC-QC')
trecqc = trecqa
dir(trecqc)
dir(trecqc.data)
len(trecqc.data)
lentrecqc.data
trecqc.data
trecqc.data[0]
trecqc.data.keys()
len(trecqc.data['test'])
len(trecqc.data['train'])
len(trec.data['train'])
len(trec.data['test'])
df = trec.data['train']
df.shape
df.columns
df = df.to_pandas()
df.columns
trec
vars(trec)
trec.files
list_datasets()
names = _
[n for n in names if 'trec' in n.lower()]
[n for n in names if 'qa' in n.lower()]
[n for n in names if 'QA' in n]
trec.info
load_dataset?
trec = load_dataset('trec', save_infos=True)
trec.info
trec.keys()
trec['train'].info
print(trec['train'].info)
print(trec['train'].info.description)
print(trec['train'].info.license)
vars(trec['train'].info)
dir(trec['train'].info)
print(trec['train'].info.features)
print(trec['train'].info.features.label-couarse)
print(trec['train'].info.features[label-coarse])
print(trec['train'].info.features['label-coarse'])
print(trec['train'].info.features['label-fine'])
df = pd.concat(trec[k].data.to_pandas() for k in trec[k])
import pandas as pd
df = pd.concat(trec[k].data.to_pandas() for k in trec[k], axis=0)
df = pd.concat([trec[k].data.to_pandas() for k in trec[k]], axis=0)
df = pd.concat([trec[k].data.to_pandas() for k in trec], axis=0)
df.shape
trec['train'].data.shape
trec = load_dataset('trec', split='all')
trec.data.shape
trec = load_dataset('trec')
trec.data.shape
trec.keys()
trec = load_dataset('trec', split='all')
df = trec.data.to_pandas()
df.shape
dir(df.data)
dir(trec.data)
dir(trec.data.features
)
trec.data.features
trec.features
trec.features['label_coarse']
labels_course = trec.features['label-coarse']
labels_fine = trec.features['label-fine']
labels_fine
labels_course = list(trec.features['label-coarse'])
labels_course = list(trec.features['label-coarse'].names)
labels_fine = trec.features['label-fine'].names
labels_fine
labels_course = trec.features['label-coarse'].names
id2name = dict(enumerate(labels_course))
labels_coarse = list(trec.features['label-coarse'].names)
labels_fine = list(trec.features['label-fine'].names)
df.shape
df.to_csv('/home/hobs/code/nlpia2/src/nlpia2/ch08/cnn-classify-text/data/trec.csv')
df.to_csv('/home/hobs/code/tangibleai/nlpia2/src/nlpia2/ch08/cnn-classify-text/data/trec.csv')
df.columns
df = df[reversed(df.columns)]
df
