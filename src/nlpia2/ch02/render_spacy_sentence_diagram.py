import spacy
from spacy import displacy
from pathlib import Path
from itertools import chain

import os
import subprocess
# import json
import logging

log = logging.getLogger(__name__)


CODE_DIR = Path(__file__).resolve().absolute()
for i in range(4):
    if CODE_DIR.name in ['code', 'nlpia2']:
        break
    # print(f"NOT code dir: {CODE_DIR}")
    CODE_DIR = CODE_DIR.parent

BASE_DIR = CODE_DIR.parent
for i in range(4):
    if BASE_DIR.name in ['nlpia-manuscript', 'nlpia2']:
        break
    # print(f"not repo dir: {BASE_DIR}")
    BASE_DIR = BASE_DIR.parent

HOME_CODE_DIR = BASE_DIR.parent.parent
print(HOME_CODE_DIR)
assert HOME_CODE_DIR.name == 'code'
MANUSCRIPT_DIR = HOME_CODE_DIR / 'tangibleai' / 'nlpia-manuscript' / 'manuscript'
assert MANUSCRIPT_DIR.is_dir()
IMAGE_DIR = MANUSCRIPT_DIR / 'images'
assert IMAGE_DIR.is_dir()
SCRIPT_WORKING_DIR = os.getcwd()


def run(command, chdir=None):
    if chdir:
        log.info(f'Temporarily changing working directory to {chdir}')
        initial_cwd = os.getcwd()
        os.chdir(chdir)

    log.warning(f'Running: {" ".join(command)}')
    output = subprocess.run(command, capture_output=True)

    if chdir:
        os.chdir(initial_cwd)

    return {
        'stderr': output.stderr.decode("utf-8").splitlines(),
        'stdout': output.stderr.decode("utf-8").splitlines()
    }


def render_svg(filepath):
    # exit_code = subprocess.call(cmd, shell=True)  # exit_code == 0 if successful
    filepath_noext = '.'.join(str(filepath).split('.')[:-1])
    # deprecated: cmd = f'inkscape --without-gui {filepath_noext}.svg -o {filepath_noext}.png'.split()
    cmd = f'inkscape {filepath_noext}.svg -o {filepath_noext}.png'.split()
    return run(command=cmd, chdir=None)


if __name__ == '__main__':
    # print(Path(__file__).resolve().absolute())
    print()
    print('-' * 70)
    print(Path(__file__).name)
    text = ("Trust me, though, the words were on their way, and when "
            "they arrived, Liesel would hold them in her hands like "
            "the clouds, and she would wring them out, like the rain. ")
    texts = [text]
    texts.append("There's no such thing as survival of the fittest. "
                 "Survival of the most adequate, maybe. ")

    try:
        nlp = spacy.load('en_core_web_md')
    except Exception:
        spacy.cli.download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')

    docs = [nlp(txt) for txt in texts]
    sentences = list(chain([list(d.sents) for d in docs]))
    print(sentences)
    options = {
        'font': 'Arial 24',
        # 'fine_grained': True,
        'add_lemma': True,
        # 'compact': True,
        'arrow_stroke': 3,  # 2
        'word_spacing': 35,  # 45 (between arrow and word)
        'distance': 150,        # 175 (between words)
    }

    for slist in sentences:
        for s in slist:
            basename = '-'.join([w.text.lower() for w in s[:5] if len(w.text) > 2])

            html = displacy.render(s, style="dep", page=True, options=options)
            filename = basename + '.html'
            filepath = IMAGE_DIR / 'ch02' / filename
            with open(filepath, 'w') as f:
                f.write(html)

            svg = displacy.render(s, style="dep", options=options)
            filename = basename + '.svg'
            filepath = IMAGE_DIR / 'ch02' / filename
            with open(filepath, 'w') as f:
                f.write(svg)

            render_svg(filepath)

    svg = displacy.render(sentences[-1][-1], style="dep", options=options)
    filename = 'survival-of-adequate-sentence-diagram.svg'
    with open(IMAGE_DIR / 'ch02' / filename, 'w') as f:
        f.write(svg)
    print('-' * 70)
    print()


r"""
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
