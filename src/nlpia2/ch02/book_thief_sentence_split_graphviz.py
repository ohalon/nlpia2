from graphviz import Graph
import shutil
from pathlib import Path

try:
    print(f"Running {Path(__file__).resolve().absolute()}")
except NameError:
    print(f"Rendering book thief sentence diagram to svg...")

# FIXME: DUPLICATED IN OTHER _graphviz.py scripts!!!!


def find_manuscript_dirs(
        PACKAGE_NAMES=('code', 'nlpia2'),
        REPO_NAMES=('nlpia-manuscript', 'nlpia2')):

    CODE_DIR = Path(__file__).resolve().absolute()
    for i in range(4):
        if CODE_DIR.name in PACKAGE_NAMES:
            break
        CODE_DIR = CODE_DIR.parent
    print(f'CODE_DIR: {CODE_DIR}')

    BASE_DIR = CODE_DIR.parent
    for i in range(4):
        if BASE_DIR.name in REPO_NAMES:
            break
        BASE_DIR = BASE_DIR.parent
    print(f'BASE_DIR: {BASE_DIR}')

    HOME_CODE_DIR = BASE_DIR.parent.parent
    print(f'HOME_CODE_DIR: {HOME_CODE_DIR}')
    assert HOME_CODE_DIR.name in PACKAGE_NAMES
    from nlpia2.constants import MANUSCRIPT_DIR
    # MANUSCRIPT_DIR = HOME_CODE_DIR / 'tangibleai' / 'nlpia-manuscript' / 'manuscript'
    assert MANUSCRIPT_DIR.is_dir()
    IMAGE_DIR = MANUSCRIPT_DIR / 'images'
    assert IMAGE_DIR.is_dir()

    # SCRIPT_WORKING_DIR = os.getcwd()
    # print(f"SCRIPT_CWD: {SCRIPT_WORKING_DIR}")

    print(f"IMAGE_DIR: {IMAGE_DIR}")
    return MANUSCRIPT_DIR, IMAGE_DIR


MANUSCRIPT_DIR, IMAGE_DIR = find_manuscript_dirs()


# TODO: get this text from the .yml file in nlpia-manuscript/code/data/ or qary/src/qary/data/nlpia/
BOOK_THIEF_TEXT = ("Reading 'The Shoulder Shrug' between two and three o'clock each morning, "
                   "post-nightmare, or during the afternoon, in the basement.")
BOOK_THIEF_TEXT = ("Trust me, though, the words were on their way, and when "
                   "they arrived, Liesel would hold them in her hands like "
                   "the clouds, and she would wring them out, like the rain.")


def get_text_bigrams(
        text=BOOK_THIEF_TEXT, tokenizer=str.split, num_tokens=8):
    tokens = tokenizer(text)
    return list(zip(tokens[:-1], tokens[1:]))[:num_tokens]


def draw_text_tokens(
        edges, name='draw-text-tokenx-graphviz', formats=['png', 'svg'],
        image_dirpath=IMAGE_DIR):
    image_dirpath = Path(image_dirpath)
    print('edges: ')
    print(edges)
    g = Graph(name)
    g.attr(rankdir='LR')
    g.attr('node', shape='box')
    for e in edges:
        g.edge(e[0], e[1])
    for f in formats:
        destfilename = f'{name}.{f}'
        g.render(filename=name, cleanup=1, view=0, format=f)
        dest = image_dirpath / Path('ch02') / destfilename
        print('Destination path for draw_text_tokens():')
        print(dest)
        try:
            dest.resolve().absolute().unlink()
            print('overwriting existing file')
        except FileNotFoundError:
            print('creating new file')
        shutil.move(destfilename, str(dest.resolve().absolute()))
    return g

# !firefox text-NLU-vector.svg


if __name__ == '__main__':
    bigrams = get_text_bigrams(BOOK_THIEF_TEXT)
    draw_text_tokens(edges=bigrams, name='book-thief-split')
