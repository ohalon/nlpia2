from pathlib import Path
commands = []
for p in Path.cwd().glob('*.adoc'):
    # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
    toks = p.name.split()
    if len(toks) > 2:
      name = toks[0] + '-' + toks[1] + '_' + '-'.join([t for t in toks[2:] if t.strip('-')])
      print(p.name, ' --> ', name)
      if input('Enter to continue'):
          break
      commands.append(f'git mv {p.name} name')      
      print(commands[-1])
from pathlib import Path
commands = []
for p in Path.cwd().glob('*.adoc'):
    # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
    toks = p.name.split()
    if len(toks) > 2:
      name = toks[0] + '-' + toks[1] + '_' + '-'.join([t for t in toks[2:] if t.strip('-')])
      cmd = f'git mv {p.name} {name}; 
      print(cmd)
      if not input('Enter to skip'):
          continue
      commands.append(cmd)      
print('\n'.join(commands))
from pathlib import Path
commands = []
for p in Path.cwd().glob('*.adoc'):
    # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
    toks = p.name.split()
    if len(toks) > 2:
      name = toks[0] + '-' + toks[1] + '_' + '-'.join([t for t in toks[2:] if t.strip('-')])
      cmd = f'git mv {p.name} {name}' 
      print(cmd)
      if not input('Enter to skip'):
          continue
      commands.append(cmd)      
print('\n'.join(commands))
hist -o -p -f 'git rename ch 1-3 adoc files only.md'
hist -f 'git rename ch 1-3 adoc files only.py'
