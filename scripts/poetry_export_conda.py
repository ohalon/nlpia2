import re
import toml
import yaml

pip2conda = {
    'graphviz': 'python-graphviz',
    'cython': 'Cython',
    'torch': 'pytorch',
}
poetry = toml.load(open('pyproject.toml'))
deps = poetry['tool']['poetry']['dependencies']
newdeps = {}
for name, ver in deps.items():
    if isinstance(ver, str):
        condition, ver = re.match(r'([>=<^]{0,2})(.*)', ver).groups()
        print('OLD:', name, condition, ver)
        if condition == '^':
            condition = '>='
            ver = condition + ver
#         elif not condition:
#             condition = '=='
        print('NEW:', name, condition, ver)
        print()
        newdeps[name] = ver

# TODO: probably better to just .update for only changed k:v pairs
poetry['tool']['poetry']['dependencies'].update(newdeps)
toml.dump(poetry, open('pyproject.reparsed.toml', 'w'))


conda = yaml.full_load(open('scripts/environment.yml'))
condadeps = conda['dependencies']

newcondadeps = []
condadepsdict = {}
for depstr in condadeps:
    if isinstance(depstr, str):
        name, condition, ver = re.match(r'([-_a-zA-Z0-9]+)([>=<]{0,2})(.*)', depstr).groups()
        print('OLD:', name, condition + ver)
        condadepsdict[name] = condition + ver
        ver = newdeps.get(name, condition + ver).strip()
        condition = ''
        condition, ver = re.match(r'([>=<^]{0,2})(.*)', ver).groups()
        if ver and not condition:
            condition = '=='
        name = pip2conda.get(name, name)
        print('NEW:', name, condition + ver)
        print()
        newcondadeps.append(name + condition + ver)
    else:
        print('!' * 100)
        print(depstr)
        print()
        newcondadeps.append(depstr)

conda['dependencies'] = newcondadeps
yaml.dump(conda, open('scripts/environment.new.yml', 'w'))


# Now go the other way (conda->poetry)
poetry = toml.load(open('pyproject.toml'))
deps = poetry['tool']['poetry']['dependencies']
newdeps = {}
for name, ver in deps.items():
    if isinstance(ver, str):
        condition, ver = re.match(r'([>=<^]{0,2})(.*)', ver).groups()
        print('  POETRY:', name, condition, ver)
        if condition == '^':
            condition = '>='
            ver = condition + ver
#         elif not condition:
#             condition = '=='
        if name in condadepsdict:
            print('CONDA:', name, condadepsdict[name])
            print('NEWPOETRY:', name, condadepsdict[name])
            print()
        newdeps[name] = ver
