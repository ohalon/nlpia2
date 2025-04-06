from nessvec.indexers import Index
import pandas as pd
from nessvec.files import load_embeddings
df = load_embeddings('word2vec')
small = df.iloc[:300_000]
index = Index(small)
docvector = index.get_doc_vector("This is a planet.")
nearest = index.get_nearest(docvector)
index.get('a')
index.get('a')
nearest
vector
docvector
docvector2 = index.get_doc_vector("This is a planet.", use_idf=True)
nearest2 = index.get_nearest(docvector2)
nearest2
docvector2 = index.get_doc_vector("This is a cold planet.", use_idf=True)
nearest2 = index.get_nearest(docvector2)
nearest2
docvector2 = index.get_doc_vector("This is a far away planet at edge of solar system cold planet.", use_idf=True)
nearest2 = index.get_nearest(docvector2)
nearest2
docvector2 = index.get_doc_vector("This is a far away planet at edge of solar system cold planet asteroid.", use_idf=True)
nearest2 = index.get_nearest(docvector2)
nearest2
hist -o -p -f ch06_extreme_summarization_word2vec_index_use_idf_planet_pluto_asteroid.hist.md
hist -f ch06_extreme_summarization_word2vec_index_use_idf_planet_pluto_asteroid.hist.py
