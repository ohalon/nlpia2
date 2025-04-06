""" Parse requirements.txt and pyproject.toml and move versions to pyproject.toml """
from pathlib import Path
import re
import sys
import toml

def get_requirement_versions(path='requirements.txt'):
    """ Read requirements.txt file and return dict of package versions """
    path = Path(path or '')
    if path.is_dir():
        path = next(iter(path.glob('**/requirements.txt')))
    reqdict = {}
    text = Path(path).open().read()
    for line in text.splitlines():
        if line.strip():
            match = re.match(r'([-_a-zA-Z0-9]+)\s*([ >=<~^,.rabc0-9]+)\s*', line)
            if match:
                name, ver = match.groups()
                reqdict[name] = ver
    return reqdict  


def normalize_name(name):
    return str(name).strip().replace('_', '-').replace(' ', '-').lower()


def pin_versions(pyproject='pyproject.toml', reqdict=None, overwrite=False):
    if not reqdict or isinstance(reqdict, (str, Path)):
        reqdict = get_requirement_versions(path=reqdict)
    reqdict = {
        normalize_name(k): v for (k, v) in 
        reqdict.items()
        }
    
    pyproj = toml.load(pyproject)
    depdict = pyproj.get('tool', {}).get('poetry', {}).get('dependencies', {})
    depdict = {
        normalize_name(k): v for (k, v) in 
        depdict.items()
        }
    
    for name, spec in reqdict.items():
        if name in depdict:
            ver = depdict[name]
            if isinstance(ver, str) and (overwrite or ver == '*'):
                depdict[name] = spec

    pyproj['tool']['poetry']['dependencies'] = depdict
    overwrite = overwrite or (input(f'Overwrite {pyproject}?')[0].lower() == 'y')
    if overwrite:
        with open(pyproject, 'w') as stream:
            toml.dump(pyproj, stream)
    return pyproj


if __name__ == '__main__':
    path = 'requirements.txt'
    if sys.argv[1:]:
        path = sys.argv[1]
    pyproj = pin_versions(reqdict=path)
    print(toml.dumps(pyproj))

