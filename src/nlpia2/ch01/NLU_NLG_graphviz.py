import os
from graphviz import Digraph
import shutil
from pathlib import Path


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

print('IMAGE_DIR:')
print(IMAGE_DIR)


def draw_nlu_diagram(name='text-NLU-vector-graphviz', center_name='NLU\n(rules, patterns or encoder)'):
    g = Digraph(name, engine='dot')  # dot neato fdp sfdp
    g.attr(rankdir='LR')
    g.attr('node', shape='box')
    g.node(center_name)
    g.attr('node', shape='plaintext')
    g.edge('Text\n(natural language)', center_name)
    g.edge(center_name, 'Vector\n(numbers, tensor, embedding, encoding)')
    # g.save()

    g.render(filename=name, cleanup=1, view=0, format='png')

    dest = IMAGE_DIR / Path('ch01') / (name + '.png')

    print('Destination NLU diagram path:')
    print(dest)
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


def draw_nlg_diagram(name='vector-NLG-text-graphviz', center_name='NLG\n(rules, templates or decoder)'):
    g = Digraph(name)
    g.attr(rankdir='LR')
    g.attr('node', shape='box')
    g.node(center_name)
    g.attr('node', shape='plaintext')
    g.edge('Vector\n(numbers, tensor, embedding, encoding)', center_name)
    g.edge(center_name, 'Text\n(natural language)')

    # Creates "vector-NLG-text-graphviz.gv" file in working directory (.dot format graph)
    # g.save()

    g.render(filename=name, cleanup=1, view=0, format='png')

    # g.render(filename=name, cleanup=1, view=0, format='pdf')
    # g.render(filename=name, cleanup=1, view=0, format='svg')

    dest = IMAGE_DIR / Path('ch01') / (name + '.png')
    try:
        dest.resolve().absolute().unlink()
    except FileNotFoundError:
        pass
    shutil.move(name + '.png', str(dest.resolve().absolute()))

    print('NLG diagram destination path:')
    print(dest)
    return g


if __name__ == '__main__':
    draw_nlu_diagram()
    draw_nlg_diagram()
