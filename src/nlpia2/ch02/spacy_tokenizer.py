import spacy
from spacy import displacy

BASE_DIR = Path(__file__).resolve().absolute().parent.parent.parent
IMAGE_DIR = BASE_DIR / 'manuscript' / 'images'

"""
>>> text = ("Trust me, though, the words were on their way, and when "
...         "they arrived, Liesel would hold them in her hands like "
...         "the clouds, and she would wring them out, like the rain.")
>>> tokens = text.split()
>>> tokens[:8]
['Trust', 'me,', 'though,', 'the', 'words', 'were', 'on', 'their']
>>> import re
>>> pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
>>> texts = [text]
>>> texts.append("There's no such thing as survival of the fittest."
...              "Survival of the most adequate, maybe.")
>>> tokens = list(re.findall(pattern, texts[-1]))
>>> tokens[:8]
["There's", 'no', 'such', 'thing', 'as', 'survival', 'of', 'the']
>>> tokens[8:16]
['fittest', '.', 'Survival', 'of', 'the', 'most', 'adequate', ',']
>>> tokens[16:]
['maybe', '.']
>>> import numpy as np  # <1>
>>> vocab = sorted(set(tokens))  # <2>
>>> ' '.join(vocab[:12])  # <3>
", . Survival There's adequate as fittest maybe most no of such"
>>> num_tokens = len(tokens)
>>> num_tokens
18
>>> vocab_size = len(vocab)
>>> vocab_size
15
texts
>>> import re
>>> pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
>>> texts = [text]
>>> texts.append("There's no such thing as survival of the fittest. "
...              "Survival of the most adequate, maybe.")
"""

text = ("Trust me, though, the words were on their way, and when "
        "they arrived, Liesel would hold them in her hands like "
        "the clouds, and she would wring them out, like the rain.")
texts = [text]
texts.append("There's no such thing as survival of the fittest."
             "Survival of the most adequate, maybe.")
spacy.cli.download('en_core_web_sm')
spacy.cli.download('en_core_web_md')
spacy.cli.download('en_core_web_lg')
nlp = spacy.load('en_core_web_md')

docs = [nlp(txt) for txt in texts]
doc = docs[-1]
sentence_spans = list(doc.sents)
sentences = list(doc.sents)
svg = displacy.render(sentences[-1], style="dep")
with open('manuscript/images/ch02/survival-of-adequate-sentence-diagram.svg', 'w') as f:
    f.write(html)
html = displacy.render(sentences[-1], style="dep", page=True)
with open('manuscript/images/ch02/survival-of-adequate-sentence-diagram.html', 'w') as f:
    f.write(html)
