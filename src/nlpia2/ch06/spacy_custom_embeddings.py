""" Load custom word vectors into gensim language model (`nlp`)

### References
- https://stackoverflow.com/a/50091148/623735
"""

import spacy

# Path to google news vectors

google_news_path = "GoogleNews-vectors-negative300.bin.gz"

# Init blank english spacy nlp object
nlp = spacy.blank('en')

# Loop through range of all indexes, get words associated with each index.
# The words in the keys list will correspond to the order of the google embed matrix
keys = []
for idx in range(3000000):
    keys.append(model.index2word[idx])

# Set the vectors for our nlp object to the google news vectors
nlp.vocab.vectors = spacy.vocab.Vectors(data=model.syn0, keys=keys)

"""
>>> nlp.vocab.vectors.shape
(3000000, 300)
"""
