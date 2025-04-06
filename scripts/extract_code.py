#!/usr/bin/env python
from nlpia2 import constants
from nlpia2.text_processing.converters import adocs2notebooks
from pathlib import Path
import sys

# from nlpia2.text_processing.extractors import extract_code
# from nlpia2.constants import BASE_DIR as NLPIA2_BASE_DIR

try:
    # tangibleai/nlpia2/
    BASE_DIR = Path(__file__).parent.parent
except Exception:
    BASE_DIR = Path.cwd()
assert BASE_DIR.is_dir(), "not {BASE_DIR}.is_dir()"
assert BASE_DIR.name == 'nlpia2', f"BASE_DIR.name should be 'nlpia' not '{BASE_DIR.name}'"


ADOC_DIR = constants.OFFICIAL_ADOC_DIR
DEST_DIR = Path(constants.__file__).parent / 'notebooks'

assert ADOC_DIR.is_dir(), "not {ADOC_DIR}.is_dir()"

if __name__ == '__main__':
    adoc_dir = ADOC_DIR
    if len(sys.argv[1:]):
        adoc_dir = Path(sys.argv[1])
    dest_dir = DEST_DIR
    if len(sys.argv[2:]):
        dest_dir = Path(sys.argv[2])
    glob = '*.adoc'
    if len(sys.argv[3:]):
        glob = Path(sys.argv[3])
    print(f'Converting {adoc_dir}/{glob} -> {dest_dir}/*.ipynb')
    nbs = adocs2notebooks(adoc_dir=adoc_dir, dest_dir=dest_dir, glob=glob)
    print(nbs)
    # nbs = adocs2notebooks(adoc_dir, dest_dir=BASE_DIR / 'src' / 'nlpia2' / 'notebooks', glob='(CH|APP)*.adoc')
    # print(nbs)
