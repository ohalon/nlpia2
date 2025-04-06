""" DEPRECATEd: use render_adoc.py and build.sh instead

FIXME: `cat` *.adoc | `asciidoctor` >> *.html | `pandoc` >> *.xml | `xmltodict` >> *.json

Convert render all adoc files in the manuscript/adoc/ dir to html then xml then json and load them into a list of dictionaries.

```bash
mkdir -p ../xml
mkdir -p ../html


for p in *.adoc ; do
  echo "path: $p";
  asciidoctor -d book -b html5 "$p" -o ../html/
done

for p in *.html ; do
  echo "path: $p";
  pandoc -t xml -o ../xml/"$p".xml --indent=2 "$p"
done
```
"""

from pathlib import Path
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
# print(HOME_CODE_DIR)
# assert HOME_CODE_DIR.name == 'code'
MANUSCRIPT_DIR = HOME_CODE_DIR / 'tangibleai' / 'nlpia-manuscript' / 'manuscript'
# assert MANUSCRIPT_DIR.is_dir()
IMAGE_DIR = MANUSCRIPT_DIR / 'images'
# assert IMAGE_DIR.is_dir()
# SCRIPT_WORKING_DIR = os.getcwd()
