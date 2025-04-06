from spacy import cli
cli.download('en_core_web_md')
from qary.etl.wikicrawl import *
pages = {'Abundance (economics)': None}
pages2 = walk_wikipedia(pages=pages, depth=1)
pages2
pages2['Abundance (economics)'].content
pages2['Abundance (economics)'][-1]
pages
pages2 = walk_wikipedia(pages=pages, depth=2)
pages
pages2['Abundance (economics)'][-1].content
print(pages2['Abundance (economics)'][-1].content)
print(pages2['Abundance (economics)'][-1].links)
print(pages2['Abundance (economics)'][-1].__dict__)
meili = CLIENT.index('wikipedia')
meili.add_documents_json([pages[t][-1].__dict__ for t in pages])
meili.get_task(0)
meili.get_task?
meili.get_task(2)
meili.search('prosocial')
ans = meili.search('prosocial')
ans.keys()
ans['hits']
ans.keys()
ans['nbHits']
ans['query']
ans['limit']
ans['processingTime']
ans['processingTimeMs']
ans['hits'].keys()
len(ans['hits'])
ans['hits'][0].keys()
ans['hits'][0]['title']
ans = meili.search('social')
len(ans['hits'])
pages['Ecological economics']
pages['Ecological economics'][-1]
pages['Ecological economics'][-1].content
'prosocial' in pages['Ecological economics'][-1].content.lower()
'social' in pages['Ecological economics'][-1].content.lower()
from nessvec.indexers import Index
from nessvec.files import load_embeddings
load_embeddings('word2vec')
df = _
df.loc['prosocial']
w2vi = Index(df.iloc[:500_000])
w2vi.get_nearest?
w2vi.get_nearest('prosocial')
w2vi.get_nearest(df.loc['prosocial'].values)
w2vi.get_nearest??
w2vi.get_nearest('prosocial')
w2vi = Index(df.iloc[:1_000_000])
w2vi.get_nearest('prosocial')
w2vi.get_nearest('social')
w2vi['prosocial']
w2vi.keys()
w2vi.get('prosocial')
w2vi.vocab
w2vi.vocab['prosocial']
w2vi.tok2id['prosocial']
pd.Index
import pandas as pd
pd.Index
