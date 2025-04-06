# check_environment.yml
from pathlib import Path
import re
import subprocess
import sys
import yaml


def get_environment(filepath):
    filepath = Path(filepath).expanduser().resolve().absolute()
    if filepath is None:
        command = ['conda', 'env', 'export']
        # print(f'Running: {" ".join(command)}')
        completed_process = subprocess.run(command, capture_output=True)
        return yaml.full_load(completed_process.stdout)
    else:
        return yaml.full_load(filepath.open())


def get_package_names(dependencies):
    packages_installed = []
    for line in dependencies:
        if isinstance(line, str):
            try:
                packages_installed.append((re.findall(r'[-\w]+', line)[0], line))
            except (IndexError,) as e:
                print('!' * 120)
                print(e)
                print(f'Unable to find any package names in: {line}')
                print()
        else:
            print(f'Skipping: {line}')
    packages_installed = dict(packages_installed)
    return dict(packages_installed)


if __name__ == '__main__':
    env_activated = get_package_names()
    packages_installed = get_package_names(env_activated['dependencies'])

    if len(sys.argv) == 3:
        env_file_needed, env_file_here = sys.argv[1:]
    elif len(sys.argv) == 2:
        env_file_needed, env_file_here = sys.argv[1], 'environment.yml'
    else:
        print('USAGE: python scripts/check_env.py ../dest_pkg/scripts/environment.yml ../source_pkg/scripts/environment.yml')
    env_needed = get_package_names(env_file_here)

    env_file_installed = env_activated['name']
    if env_file_installed.endswith('env'):
        dirname = env_file_installed.strip()[:-3]
    else:
        dirname = env_file_installed.strip()
    env_file_installed = get_package_names(f'~/code/tangibleai/{dirname}/scripts/')

    packages_needed = get_package_names(env_needed['dependencies'])
    packages_installed_file = get_package_names(env_file_installed['dependencies'])

    for pkg in packages_needed:
        if pkg not in packages_installed:
            print(f'Missing: {pkg} from {packages_needed[pkg]}')
