import spacy
import os
from pathlib import Path
try:
    name = 'en_core_web_sm'
    nlp = spacy.load(name)
    print(f'SpaCy {name} is already installed.')
except (NameError, ImportError) as e:
    print(e)
    print(f'Downloading the **{name}** spacy language model...')
    spacy.cli.download(name)
    nlp = spacy.load(name)

modeldir = Path(nlp._path).parent.parent
files = os.listdir(str(modeldir))
# print('\n'.join(files));
name = 'en_core_web_md'
if any(f.startswith(name) for f in files):
    print(f'SpaCy {name} is installed here: {modeldir}/{name}*')
else:
    print(f"Couldn't find {name}\n  in {modeldir}\n  among {files}")
    print(f'Trying again to download **{name}** spacy language model...')
    spacy.cli.download(name)
