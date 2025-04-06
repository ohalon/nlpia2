from nessvec.indexers import Index
import pandas as pd
from nessvec.files import load_embeddings

df = load_embeddings('word2vec')
small = df.iloc[:500_000]
index = Index(small)
me = df.T['reckless_abandon']
aggro = df.T['aggressive'] + df.T['altruism']
print('reckless_abandon fearless altruism')
aggro_me = (me + aggro) / 3
print(index.get_nearest(aggro_me.values))
fearless = df.T['fearless'] / 2 + df.T['altruism'] / 2
print('fearless altruism')
print(index.get_nearest(fearless))
print("fearless altruism agressive good kindness")
print(
    index.get_nearest(index.get_doc_vector("fearless altruism agressive good kindness")))
print("fearless altruism agressive good")
print(index.get_nearest(index.get_doc_vector("fearless altruism agressive good")))
print("fearless generosity agressive good")
print(index.get_nearest(index.get_doc_vector("fearless generosity aggressive good", use_idf=False)))
