""" Sense2vec (2015 paper) creates better embeddings, especially for noun-phrases

### References
- https://github.com/explosion/sense2vec
- [sense2vec - A Fast and Accurate Method for Word Sense Disambiguation In Neural Word Embeddings](
https://arxiv.org/abs/1511.06388 by Andrew Trask, Phil Michalak, John Liu)
"""

from sense2vec import Sense2Vec

s2v = Sense2Vec().from_disk("/path/to/s2v_reddit_2015_md")
query = "natural_language_processing|NOUN"
assert query in s2v
vector = s2v[query]
freq = s2v.get_freq(query)
most_similar = s2v.most_similar(query, n=3)
# [('machine_learning|NOUN', 0.8986967),
#  ('computer_vision|NOUN', 0.8636297),
#  ('deep_learning|NOUN', 0.8573361)]
