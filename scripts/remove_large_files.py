# remove large files
# 
import os
from tqdm import tqdm

cmd = """
git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| sort --numeric-sort --key=2 \
| cut -c 1-12,41- | \
$(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
"""

stdout_lines = []
with os.popen(cmd) as stream:
    table = []
    for line in tqdm(stream.readlines()):
        row = [line[12:20].strip(), line[20:].strip()]
        table.append(row)



MAX_MB = 25

# cmd_fstring = "git filter-branch --index-filter 'git rm -rf --cached --ignore-unmatch {path}' HEAD"
cmd_fstring = "git filter-repo --invert-paths --path-match {path}"
for size, path in table:
    if size.endswith('MiB'):
        size = float(size[:-3])
        if size >= MAX_MB:
            print(f"Removing {size}MB file: {path}") 
            cmd = cmd_fstring.format(path=path)
            with os.popen(cmd) as stream:
                for line in stream.readlines():
                    print(line)


