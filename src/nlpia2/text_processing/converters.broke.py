""" Utilities for manipulating asccidoc (asciidoctor) documents

Typical DocTestParser Expression objects:
{'source': 'import spacy\n',
 'want': '',
 'lineno': 64,
 'indent': 0,
 'options': {},
 'exc_msg': None},
{'source': 'nlp = spacy.load("en_core_web_sm")\n',
 'want': '',
 'lineno': 65,
 'indent': 0,
 'options': {},
 'exc_msg': None},
{'source': 'sentence = \"\"\"The faster Harry got to the store, the faster Harry,\n    the ...',
 'want': '',
 'lineno': 67,
 'indent': 0,
 'options': {},
 'exc_msg': None}
"""
from doctest import DocTestParser
import logging
from pathlib import Path
import re
import sys

import nbformat as nbf
from tqdm import tqdm

from nlpia2.constants import OFFICIAL_ADOC_DIR  # , ADOC_DIR
from nlpia2.text_processing.extractors import parse_args
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output


log = logging.getLogger(__name__)
HEADER_TEXT = f"""\
# Imports and Settings

>>> import pandas as pd
>>> pd.options.display.max_columns = 3000
"""
HEADER_BLOCKS = [
    dict(source='### Imports and Settings', typ='markdown'),
    dict(source='''\
        >>> import pandas as pd
        >>> pd.options.display.max_columns = 3000
        ''', typ='python')
]

TEST_TEXT = f"""\
# A dataframe

>>> import pandas as pd
>>> pd.options.display.max_columns = 3000
>>> pd.DataFrame([[1,2],[3,4]])
   0  1
0  1  2
1  2  3
"""
TEST_BLOCKS = [
    dict(source='### A dataframe', typ='markdown'),
    dict(source='''\
        >>> pd.DataFrame([[1,2],[3,4]])
        ''', typ='python',
         want='''\
           0  1
        0  1  2
        1  2  3
        ''')
]

# use newlines and groups instead of list of patterns
CODEHEADER = [
    r'^([.][- \w\t\(\)\[\]\|\*\'"!@#{}&^%$+=<,>.?/]+[ \t]*)?[ \t]*$',  # '.Problem setup'
    r'^\[source,[ \t]*python\][ \t]*$',
    r'^----[-]*[ \t]*$',
]
MULTILINE_CODEHEADER = '\n'.join([ln.rstrip('$') for ln in CODEHEADER])

# Need multiline pattern to capture multiple annotations and blank line after annotations
CODEFOOTER = r'\n(----[-]*\s*\n(<\d+>\s?[^\n]+\n)*)'


def re_matches(patterns, lines, newline='\n', match_objects=True):
    r""" Match multiple regular expressions to consecutive lines of text

    >>> patterns = '^([.][\\w\\s]+)?\s*$\n^\[source\]$\n^[-]+$'.split('\n')
    >>> lines = '----\n.hello\n\n[source]\n----\nworld\n----'.split('\n')
    >>> re_matches(patterns, lines)
    []
    >>> re_matches(patterns, lines[2:])
    [<re.Match object; span=(0, 0), match=''>,
     <re.Match object; span=(0, 8), match='[source]'>,
     <re.Match object; span=(0, 4), match='----'>]
    >>> patterns = r'\n(----[-]*\s*\n((<\d+>\s?[^\n]+)?\n)+)'
    >>> lines = '\n----\n.hello\n\n[source]\n----\nworld\n----\n<1> annot.\n\n'
    >>> re_matches(patterns, lines)

    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    if isinstance(patterns, str):
        it = re.finditer(patterns, newline.join(lines))
        try:
            return next(it).group().splitlines()  # footers can't return a list of match objects
        except StopIteration:
            return []
    matches = []
    for pat, text in zip(patterns, lines):
        match = re.match(pat, text)
        if not match:
            return matches
        matches.append(match)
    if not match_objects:
        return [m.group() for m in matches]
    return matches


def re_matches_multiline(patterns, lines, newline='\n', match_objects=True):
    r""" Match multiple regular expressions to consecutive lines of text

    >>> patterns = '^([.][\\w\\s]+)?\\s*$\n^\[source\]$\n^[-]+$'.split('\n')
    >>> lines = '----\n.hello\n\n[source]\n----\nworld\n----'.split('\n')
    >>> re_matches(patterns, lines)
    []
    >>> re_matches(patterns, lines[2:])
    [<re.Match object; span=(0, 0), match=''>,
     <re.Match object; span=(0, 8), match='[source]'>,
     <re.Match object; span=(0, 4), match='----'>]
    >>> patterns = r'\n(----[-]*\s*\n((<\d+>\s?[^\n]+)?\n)+)'
    >>> lines = '\n----\n.hello\n\n[source]\n----\nworld\n----\n<1> annot.\n\n'
    >>> re_matches(patterns, lines)

    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = newline.join([ln.rstrip() for ln in lines])
    matches = []
    for pat, text in zip(patterns, lines):
        match = re.match(pat, text)
        if not match:
            return matches
        matches.append(match)
    if not match_objects:
        return [m.group() for m in matches]
    return matches


def get_examples(text, join_consecutive=True):
    r""" Extract all doctest code and output examples from asciidoc (adoc) text

    >>> text = ">>> import hello\n>>> hello.world()\n'hi'\n"
    >>> get_examples(text)[0]['source']
    'import hello\nhello.world()'
    >>> get_examples(text)[0]['want']
    "'hi'\n"
    """
    dtparser = DocTestParser()
    try:
        examples = dtparser.get_examples(text)
    except ValueError:
        msg = f'Error processing doctests in {text[:40]}...{text[-40:]}'
        log.error(msg)
        raise
    if not join_consecutive:
        return examples

    enum, bnum = 0, 0
    blocks = [[]]
    while enum < len(examples):
        ex = vars(examples[enum])
        blocks[bnum].append(ex)
        if ex['want']:
            blocks.append([])
            bnum += 1
        enum += 1
    examples = []
    for lines in blocks:
        if not lines:
            continue
        ex = lines[0].copy()
        ex['source'] = '\n'.join([rstrip_one_newline(line['source']) for line in lines])
        ex['want'] = lines[-1]['want']
        examples.append(ex)
    return examples


def rstrip_one_newline(text, newlines='\r\n'):
    """Like text.rstrip('\n').rstrip('\r') but only strips a maximum of 1 character with each strip """
    for c in reversed(newlines):
        if text.endswith(c):
            text = text[:-1]
    return text


# def get_headings(text):
#     headings = []
#     for lineno, line in enumerate(text.splitlines()):
#         match = re.match(r'^[=]{1,5}[ ]{0,2}.+', line)
#         if match:
#             headings.append(dict(
#                 lineno=i,
#             ))


def get_code_blocks(text,
                    header_patterns=CODEHEADER,
                    footer_patterns=None,  # CODEFOOTER,
                    min_header_len=2,
                    min_footer_len=2,
                    max_block_lines=64):
    if footer_patterns is None:
        footer_patterns = [header_patterns[-1]]
    lines = text.splitlines()
    blocks = []
    i = len(header_patterns)
    while i < len(lines) - 1:
        header_matches = re_matches(header_patterns, lines[i:])
        # some number of header lines must match (2 for adoc, [source,python]\n----\n)
        if not len(header_matches) >= min_header_len:
            i += 1
            continue
        line_number = i
        if header_matches[0].group().strip():
            line_number -= 1
        i += len(header_matches)
        # last line of header must match
        if not re.match(header_patterns[-1], header_matches[-1].group()):
            i += 1
            continue
        block = []
        # TODO: include docutils examples(parsed code and output examples)
        while i < len(lines) - min_footer_len and len(block) < max_block_lines:
            block.append(lines[i])
            i += 1
            footer_matches = re_matches(footer_patterns, lines[i:])
            # if len(footer_matches) >= len(footer_patterns):
            #     i += len(footer_matches)
            #     break
        block = dict(
            preceding_text=lines[line_number - 1],
            preceding_blank_line=lines[line_number],
            line_number=line_number,
            header='\n'.join([m.group() for m in header_matches]),
            prompted_source='\n'.join([s.rstrip('\n') for s in block]),
            footer='\n'.join([m.group() for m in footer_matches]),
            following_blank_line=lines[i],
        )
        if i + 1 < len(lines):
            block['following_text'] = lines[i + 1]
        examples = get_examples(block['prompted_source'], join_consecutive=True)
        if examples:
            for example in examples:
                if 'source' not in example:
                    continue
                newblock = block.copy()
                # FIXME: also update line_number from examples
                newblock.update(dict(source=example['source'], want=example['want']))
                blocks.append(newblock)
        else:
            block['source'] = block['prompted_source']
            block['want'] = ''
            blocks.append(block)
    return blocks


def adoc_doctests2ipynb(adocs=Path('.'), dest_filepath=None, **kwargs):
    adocs = Path(adocs)
    text = kwargs.pop('text', None) or ''
    if adocs.is_file():
        text = text + '\n' + adocs.read()
    dest_filepath = dest_filepath if not dest_filepath else Path(dest_filepath)
    examples = get_examples(text)

    nb = new_notebook()
    cells = []
    cells.append(new_markdown_cell(f"#### {adocs}"))

    for exp in examples:
        # need to run the doctest parser on a lot of text to get attr names right
        if isinstance(exp, str):
            cells.append(new_markdown_cell(exp))
        if 'text' in exp:
            cells.append(new_markdown_cell(exp['text']))
        if 'prompted_source' in exp:
            new_code_cell(exp['prompted_source'])

    nb['cells'] = cells
    if dest_filepath:
        with dest_filepath.open('w') as f:
            nbf.write(nb, f)
    return nb


def find_title(text, pattern=r'^\s*[=#]\s?(.+)$'):
    r""" First first line that matches pattern (r'^\s*[=#]\s?.+$')"""
    for line in text.splitlines():
        if re.match(pattern, line):
            return line
    for line in text.splitlines():
        if line.strip():
            return line


def adoc2ipynb_file(filepath=None, dest_filepath=None, text=None, prompt=False, output=False):
    """ Extract code blocks and their captions from a SINGLE adoc file and save to an *.ipynb file"""
    try:
        text = Path(filepath).open().read()
    except (TypeError, OSError, IOError, FileNotFoundError) as e:
        log.error(f'Invalid filepath: {filepath}\n  ERROR: {e}')

    dest_filepath = None if not dest_filepath else Path(dest_filepath)
    blocks = get_code_blocks(text)

    nb = new_notebook()
    cells = []
    title = find_title(text)
    if filepath:
        title = f'[`{filepath.with_suffix("").name}`]({filepath})'
    if filepath:
        cells.append(new_markdown_cell(f"#### {title}"))

    print(len(blocks), filepath)
    print(dest_filepath)
    for block in blocks:
        # need to run the doctest parser on a lot of text to get attr names right
        if 'source' not in block:
            continue
        if len(block['header'].splitlines()) == 3:
            cells.append(new_markdown_cell('#### ' + block['header'].splitlines()[0]))
        if prompt:
            cells.append(new_code_cell(block['prompted_source']))
        else:
            cells.append(new_code_cell(block['source']))
        if output:
            cells.append(new_output(block['output']))

    nb['cells'] = cells
    if dest_filepath:
        with dest_filepath.open('w') as f:
            nbf.write(nb, f)
    return nb


# def adocs2outlines(adoc_dir=Path('.'), dest_dir=None, glob='Chapter-*.adoc'):


def adocs2notebooks(adoc_dir=Path('.'), dest_dir=None, glob='Chapter-*.adoc'):
    """ Convert a directory containing MANY adoc files into jupyter notebooks

    Inputs:
      adoc_dir: Path or str to directory *.adoc files containing code blocks
      dest_dir: Path or str to directory where *.ipynb should be saved
      glob: glob pattern to match adoc filenames, default='Chapter-*.adoc'

    Returns:
      list of dicts: [{
        text: json text string,
        nb: v4 Notebook object (nbformat.notebooknode.NotebookNode)
        filepath: Path object containing location where notebook saved in dest_dir
        }, ...]
    """
    adoc_dir = Path(adoc_dir or OFFICIAL_ADOC_DIR)
    if not adoc_dir.is_dir() or len(list(adoc_dir.glob(glob))) < 12:
        adoc_dir = Path('.')
    outlines = []
    if not dest_dir:
        dest_dir = adoc_dir.parent / 'notebooks'
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)
    for adoc_filepath in tqdm(list(adoc_dir.glob(glob))):
        dest_filepath = dest_dir / adoc_filepath.with_suffix('.ipynb').name
        outlines.append(
            adoc2outline_file(filepath=adoc_filepath, dest_filepath=dest_filepath)
        )
    return outlines


def convert(format='ipynb', **kwargs):
    """ Convert files in adocs to dictionary of notebook json (text), notebook object (nb), and filepath """
    filepath = kwargs.pop('adocs', kwargs.pop('adoc', kwargs.pop('filepath')))
    if filepath:
        print(filepath)
        text = Path(filepath).open().read()
        print(len(text))
    else:
        text = TEST_TEXT
    text = HEADER_TEXT + '\n\n' + text
    return dict(nb=adoc2ipynb_file(text=text), text=text, filepath=filepath)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        kwargs = parse_args(
            format='ipynb',
            description='Convert adoc code blocks and preceding heading text to a jupyter notebook',
            adocs_help='File path to input adoc file',
            output_help='File path to output ipynb file')
        # format = kwargs.pop('format')
        results = convert(**kwargs)
        # print(results)
