""" Based on the make_outline.sh script -- create detailed outline book as a markdown file "outline.md"

Only uses the chapter titles and headings
"""

from pathlib import Path
import re
import sys

HEADING_SYMBOL_LEVELS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
MAX_HEADING_SYMBOLS = max(HEADING_SYMBOL_LEVELS)
MIN_HEADING_SYMBOLS = min(HEADING_SYMBOL_LEVELS)

RE_HEADING = r'^([=]+\s*|:Chapter:\s?[0-9]+)(.+)$'

RE_HEADING = r'^([=]{'
RE_HEADING += str(MIN_HEADING_SYMBOLS) + ',' + str(MAX_HEADING_SYMBOLS)
RE_HEADING += r'}\s?|:Chapter:\s?[0-9]+)(.+)$'
CRE_HEADING = re.compile(RE_HEADING)


SYM_CHAPTER_TITLE = '= '  # "= Reasoning with word embeddings"
SYM_CHAPTER_NUM = ':Chapter:'  # ":Chapter: 6"


try:
    BASE_DIR = Path(__file__).parent.parent.parent
except Exception:
    BASE_DIR = Path.cwd()
assert BASE_DIR.is_dir(), "not {BASE_DIR}.is_dir()"
assert BASE_DIR.name == 'nlpia2'

ADOC_DIR = BASE_DIR.parent / 'hobs' / 'nlpia-manuscript' / 'manuscript' / 'adoc'
assert ADOC_DIR.is_dir(), "not {ADOC_DIR}.is_dir()"


def make_path(p):
    """ Should be `Path()`, perhaps or should monkey-patch the Path class """
    return Path(p).expanduser().resolve().absolute()


def open_path(filepath, *args, **kwargs):
    return make_path(filepath).open(*args, **kwargs)


def extract_adoc_outline(filepath):
    outline_lines = []

    with open_path(filepath, 'rt') as fin:
        for line in fin:
            groups = CRE_HEADING.match(line)
            if groups and groups[1] and groups[2]:
                if groups[1].startswith('='):
                    level = len(groups[1].strip()) - 1
                    text = groups[2]
                    newlines = '\n' * int(level == 0)
                    outline_lines.append('  ' * level + newlines + "* " + text)
                elif groups[2].lower().startswith(SYM_CHAPTER_TITLE.lower()):
                    chapter_num = groups[2]
                    outline_lines.append('* ' + groups[1] + chapter_num)

    return outline_lines


def generate_book_outline(adoc_path, glob='Chapter*.adoc'):
    adoc_path = make_path(adoc_path)
    if adoc_path.is_file():
        adoc_paths = [adoc_path]
    else:
        adoc_paths = adoc_path.glob(glob)
    for p in adoc_paths:
        yield {str(p.with_suffix('').name): extract_adoc_outline(p)}


"""
$ make_outline.sh

```python
python -c '\
  with open("docs/headings.adoc") as fin: \
    lines = fin.readlines()
  ch_lines = []
  for i, line in enumerate(lines):
    if line.lower().startswith(':chapter:'):
      toks = line[-1].split()
      ch_lines[-1] = ' '.join(toks[:1] + [line.strip()] + toks[1:])
    else:
      ch_lines.append(line)
  with open("docs/headings.adoc", 'w') as fout: \
    fout.writeines(lines)
  '
```

```bash
newline =$'\n'


sed 's/=/#/g' docs / headings.adoc > docs / headings.md
sed 's/==== /      * /g' docs / headings.adoc > docs / outline.md
sed - i 's/=== /    * /g' docs / outline.md
sed - i 's/== /  * /g' docs / outline.md
sed - i 's/= /'"\\${newline}"'* /g' docs / outline.md

# rm -f headings.adoc

# pandoc --atx-headers     --verbose     --wrap=none     --toc     --reference-links     -s -o outline.adoc     outline.md
"""

if __name__ == '__main__':
    raise NotImplementedError('This make_outline.py script is BORKED!')
    adoc_path = ADOC_DIR
    glob = 'Chapter*.adoc'
    if len(sys.argv) > 1:
        adoc_path = sys.argv[1]
    if len(sys.argv) > 2:
        glob = sys.argv[2]
    chapter_outlines = list(generate_book_outline(adoc_path, glob=glob))
    print(chapter_outlines)
