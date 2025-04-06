import argparse
import os
import copy
from graphviz import Digraph, Graph
import yaml
import sys
from pathlib import Path
import shutil
import logging
from nlpia2.constants import DATA_DIR

__version__ = None
try:
    from nlpia2 import __version__  # noqa
except ImportError:
    pass
try:
    from nlpia2.constants import __version__  # noqa
except ImportError:
    pass


log = logging.getLogger(__name__)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """


CLASSES = dict(digraph=Digraph, graph=Graph)

# __file__ defined so script can be %run within ipython
SCRIPT_WORKING_DIR = os.getcwd()
try:
    __file__ = __file__
except NameError:
    __file__ = SCRIPT_WORKING_DIR / f'{__name__}.py'


def find_code_dir(filepath=__file__):
    CODE_DIR = Path(filepath).resolve().absolute()
    for i in range(4):
        if not CODE_DIR or not CODE_DIR.name or CODE_DIR.name in ['code', 'nlpia2']:
            break
        # print(f"NOT code dir: {CODE_DIR}")
        CODE_DIR = CODE_DIR.parent
    return CODE_DIR


CODE_DIR = find_code_dir()


def find_repo_dir(dirpath=CODE_DIR):
    BASE_DIR = dirpath.parent
    for i in range(4):
        if BASE_DIR.name in ['nlpia-manuscript', 'nlpia2']:
            break
        # print(f"not repo dir: {BASE_DIR}")
        BASE_DIR = BASE_DIR.parent
    return BASE_DIR


BASE_DIR = find_repo_dir()


def find_dest_dir(home_code_dir=BASE_DIR.parent.parent):
    print('find_dest')
    print(f'home_code_dir: {home_code_dir}')
    assert home_code_dir.name == 'code'
    MANUSCRIPT_DIR = home_code_dir / 'tangibleai' / 'nlpia-manuscript' / 'manuscript'
    assert MANUSCRIPT_DIR.is_dir()
    dest_dir = MANUSCRIPT_DIR / 'images'
    assert dest_dir.is_dir()
    print(f'dest_dir: {dest_dir}')
    return dest_dir


DEST_DIR = find_dest_dir()


def wrap_text(text, max_line_width=10):
    text = str(text)
    lines = []
    words = text.split()
    if len(text) < max_line_width:
        return text
    for i, w in enumerate(words):
        if len(w) < 3:
            if i:
                lines[-1] += f' {w}'
            else:
                if len(words) > 1:
                    words[i + 1] = f'{w} ' + words[i + 1]
                else:
                    lines.append(w)
        else:
            lines.append(w)
    return '\n'.join(lines)


def load_graphviz(filepath=None, engine='sfdp', attr=None, node_attr=None):
    """ Load yaml file into dictionary of graphviz args for pygraphviz

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
    filepath = filepath or CODE_DIR / 'data' / 'nlp-applications-graphviz.yml'
    engine = engine or 'sfdp'  # neato, fdp, sfdp, dot
    # u = Digraph('unix', filename='unix.gv',
    #             node_attr={'color': 'lightblue2', 'style': 'filled'})
    ATTR = dict(
        # engine='fdp',
        rankdir='LR',
        # layout="neato",
        # size='6,6',
        #     nodesep=1,
        #     ranksep=1,
    )
    attr = attr or ATTR

    node_attr = node_attr or dict(shape='plaintext')

    filepath = str(filepath)
    with open(filepath) as fin:
        y = yaml.full_load(fin)
    attr = copy.deepcopy(dict(attr))
    node_attr = copy.deepcopy(node_attr)
    log.warning(f'yaml filepath: {filepath}')
    Klass = CLASSES[y.get('class', 'digraph').lower()]
    name = ''.join(str(filepath).split('.')[:-1])
    name = str(y.get('name') or name)
    engine = str(y.get('engine') or engine)
    log.warning(f'engine: {engine}')
    attr.update(y.get('attr', {}))
    node_attr.update(y.get('node_attr', {}))
    g = Klass(name, filename=name + '.gv',
              engine=engine, node_attr=node_attr)
    g.attr(**attr)
    # print(g)
    for i, node in enumerate(y.get('nodes', [])):
        label, kwargs = None, {}
        if isinstance(node, str):
            label = node
        elif len(node) == 1:
            label = str(node[0])
        elif len(node) == 2:
            if isinstance(node[1], dict):
                label = str(node[0])
                kwargs = dict(node[1])
            else:
                log.warning(f"Unable to parse node #{i}: {node}")
        if label is not None:
            # if label is just whitespace, then don't give it a box/oval/circle shape
            if not label.strip():
                kwargs.update({'shape': 'plaintext'})
            log.warning(f"{label}, {kwargs}")
            if len(kwargs):
                g.node(wrap_text(label), **kwargs)
            else:
                g.node(wrap_text(label))

    unique_edges = set()
    for e in y.get('edges', []):
        edge_str = str(e)
        if edge_str in unique_edges:
            continue
        unique_edges.add(edge_str)
        # print(e)
        if len(e) == 2:
            g.edge(wrap_text(e[0]), wrap_text(e[1]))
        elif len(e) == 3:
            if isinstance(e[2], str):
                e[2] = dict(label=wrap_text(e[2]))
            g.edge(wrap_text(e[0]), wrap_text(e[1]), **e[2])
    return g


def brittle_parse_args(args=None):
    """ Assumes 3 cli args are YAML_FILE CHNUM DEST_DIR  """
    args = args or sys.argv[1:]
    dest_dir = DEST_DIR if len(args) < 3 else Path(args[2])
    assert dest_dir.is_dir()
    chnum = '' if len(args) < 2 else str(args[1])
    dest_dir /= str(chnum)
    assert dest_dir.is_dir()
    yaml_filepath = None if len(args) < 1 else Path(args[0]).expanduser().resolve().absolute()
    print(f'yaml_filepath: {yaml_filepath}')
    return dict(dest_dir=dest_dir, yaml_filepath=yaml_filepath, loglevel=logging.WARNING)


def parse_args(args=None):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      argparse.Namespace: command line parameters as attributes of namespace object
    """
    parser = argparse.ArgumentParser(
        description="Transpile yaml->graphviz.dot then render svg/png files")
    parser.add_argument(
        "--version",
        action="version",
        version="yaml_graphviz {ver}".format(ver=__version__))
    parser.add_argument(
        dest="yaml_file",
        nargs='?',
        help="Path to yaml file containing graph specification for graphviz diagram",
        type=Path,
        metavar="YAML_FILE")
    parser.add_argument(
        dest="dest_subdir",
        nargs='?',
        help="Subdirectory within dest_dir (e.g. `ch05`)",
        type=Path,
        metavar="CHXX")
    parser.add_argument(
        '-d',
        '--dest',
        dest="dest_dir",
        help="images directory to render svg and png to",
        default=Path(DEST_DIR),
        type=Path,
        metavar="DEST_DIR")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    args = sys.argv[1:] if args is None else args
    args = parser.parse_args(args)
    return dict(yaml_filepath=args.yaml_file,
                dest_dir=args.dest_dir,
                loglevel=args.loglevel)


def render_yaml_graphviz(yaml_filepath=None, dest_dir=DEST_DIR):
    if yaml_filepath is None:
        yaml_filepath = next(DATA_DIR.glob('*graphviz.yml'))
    assert yaml_filepath.is_file()
    g = load_graphviz(filepath=yaml_filepath)
    name = str(g.name)
    for ext in ['svg', 'png']:
        g.render(filename=name, cleanup=1, view=0, format=ext)
        # g.save()
        # g.view()

        dest = DEST_DIR / (name + '.' + ext)
        log.warning(f'svg filepath: {dest}')
        try:
            dest.resolve().absolute().unlink()
        except FileNotFoundError:
            pass
        shutil.move(name + '.' + ext, str(dest.resolve().absolute()))
    # g.view()


if __name__ == '__main__':
    # FIXME: kwargs = parse_args()
    kwargs = parse_args()
    setup_logging(kwargs.pop('loglevel'))
    print(kwargs)
    render_yaml_graphviz(**kwargs)
