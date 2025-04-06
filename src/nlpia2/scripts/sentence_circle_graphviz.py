import os
import graphviz
import shutil
from pathlib import Path

from nessvec.text import tokenize


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

DEFAULT_SENTENCE = "Teaching is the highest form of understanding."


def draw_circle(sentence=DEFAULT_SENTENCE, name='sentence-circle-graphviz',
                class_name='Graph', engine='circo', center_name='attention',
                image_dir=IMAGE_DIR, image_subdir=''):
    """ Draw a sentence in a circle with connections between all words.
    Inputs:
      engine (https://graphviz.org/docs/layouts/):
        dot: hierarchies that look like org charts (best for Digraphs)
        neato: minimum energy layoud (Brownian motion? Lagrange functions?)
        circo: all nodes on edge of circle
        fdp: force-directed graph that reduces spring force instead of energy (like neato), order of "appearance" out from center (first nodes)
        osage: hierarchical clusters of subgraphs
        patchwork: squarified treemap layout
        sfdp: multiscale fdp for large graphs with many nodes
    """
    print(f'name: {name}')
    print(f'sentence: {sentence}')
    tokens = tokenize(sentence)
    print(f'tokens: {tokens}')
    print(f'image_dir: {image_dir}')
    dest = image_dir / Path(image_subdir) / (name + '.png')

    print('Destination circle network diagram path:')
    print(dest)

    g = getattr(graphviz, class_name)(name, engine=engine)  # dot neato fdp sfdp circo
    g.attr(rankdir='LR')
    g.attr('node', shape='box')
    g.node(center_name)
    g.attr('node', shape='plaintext')
    # g.edge('Text\n(natural language)', center_name)

    # g.save()

    g.render(filename=name, cleanup=1, view=0, format='png')

    try:
        dest.resolve().absolute().unlink()
    except FileNotFoundError:
        pass
    shutil.move(name + '.png', str(dest.resolve().absolute()))
    print(dest)
    return g

# g.render(filename=name, cleanup=1, view=0, format='pdf')
# g.render(filename=name, cleanup=1, view=0, format='svg')
# !firefox text-NLU-vector.svg


def draw_matrix(sentence=DEFAULT_SENTENCE, name='sentence-matrix', image_dir=IMAGE_DIR):
    print(f'name: {name}')
    print(f'sentence: {sentence}')
    tokens = tokenize(sentence)
    print(f'tokens: {tokens}')
    print(f'image_dir: {image_dir}')


if __name__ == '__main__':
    draw_circle()
    draw_matrix()
