# pip install spacy-experimental
# pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
# note spacy 3.4.0 version number (must be exactly the same for binary compatibility)
# https://github.com/explosion/spacy-experimental/releases/tag/v0.6.0
# https://github.com/explosion/spacy-experimental#coreference-components
# https://github.com/explosion/projects/tree/v3/experimental/coref
import spacy
from spacy_language_model import load
nlp = load('en_coreference_web_trf')
doc = nlp('This sentence is what it is.')
doc.clusters
doc.spans
text = "John Smith called from New York, he says it's raining in the city."
doc = nlp(text)
doc.spans
text = 'Gebru had determined that publishing research papers was more effective at bringing forth the ethical change she was focused on than pressing her superiors in the company. She and five others coauthored a research paper, "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"'
doc = nlp(text)
doc.spans
# hist -o -p -f src/nlpia2/ch11/gebru-she-coreference-resolution.hist.ipy
# hist -f src/nlpia2/ch11/gebru-she-coreference-resolution.hist.py
