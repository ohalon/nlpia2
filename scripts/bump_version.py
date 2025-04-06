#!/usr/bin/env python
from pathlib import Path
import re
import shutil

BASE_DIR = Path(__file__).parent.parent
PYPROJECT_PATH = BASE_DIR / 'pyproject.toml'
PATTERN = re.compile(r'(version\s*=\s*)[\'"]?(\d(\.\d+)+)[\'"]?\s*')

if __name__ == '__main__':
    verline = None
    with PYPROJECT_PATH.open() as fin:
        lines = []
        verline = None
        for line in fin:
            lines.append(line)
            if verline:
                continue
            match = PATTERN.match(line)
            if match:
                print(f"Found match.groups(): {dict(list(enumerate(match.groups())))}")
                ver = [int(x) for x in match.groups()[1].split(".")]
                print(f"             Old ver: {ver}")
                ver[-1] += 1
                print(f"             New ver: {ver}")
                ver = ".".join([str(x) for x in ver])
                print(f"         New ver str: {ver}")
                verline = f'version = "{ver}"\n'
                print(f"        New ver line: {verline}")
                lines[-1] = verline
                print(f"        New ver line: {lines[-1]}")

    if verline:
        shutil.copy(PYPROJECT_PATH, PYPROJECT_PATH.with_suffix('.toml.bak'))
        with PYPROJECT_PATH.open('w') as fout:
            fout.writelines(lines)
