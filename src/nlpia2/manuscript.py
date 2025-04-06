# manuscript.py
""" Tools for manuscript ADOC file manipulation for Manning """
from pathlib import Path
import shutil

from nlpia2.constants import BIGDATA_DIR, MANUSCRIPT_DIR


def translate_to_matko_path(path):
    toks = path.name.split()
    name = path.name
    if len(toks) > 2:
        name = toks[0] + '-' + toks[1] + '_' + '-'.join([t for t in toks[2:] if t.strip('-')])
        name = name.replace('(', '').replace(')', '')
    return name


def git_rename_commands(glob='Chapter ?? --*.adoc',
                        translater=translate_to_matko_path,
                        ):
    commands = []
    for p in Path.cwd().glob(glob):
        # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
        new_name = translater(p)
        cmd = f'git mv "{p.name}" "{new_name}"'
        print(f'# {cmd}')
        if not input('Enter to skip'):
            continue
        print(cmd)
        commands.append(cmd)
    return commands


def rename_path_pairs(glob='Chapter-??_*.adoc',
                      translater=translate_matko,
                      interactive=False,
                      ):
    path_pairs = []
    for p in Path.cwd().glob(glob):
        # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
        new_name = translater(p.name)
        # print(new_name)
        if new_name != p.name:
            cmd = f'git mv "{p.name}" "{new_name}"'
            if interactive:
                print(f'# {cmd}')
                if not input('Enter to skip'):
                    continue
                print(cmd)
            path_pairs.append((p, p.parent / new_name))
    return path_pairs


def translate_hobs(path):
    path = str(path).lower()
    path = path[:2] + path[len('Chapter-'):][:2] + '.adoc'
    return path


def update_nlpia2_data(src=MANUSCRIPT_DIR, dst=BIGDATA_DIR):
    path_pairs = rename_path_pairs()
    for orig_path, dest_path in path_pairs:
        shutil.copyfile(orig_path, dst=str(dst / Path(dest_path).name))
    return path_pairs


def rename_chapters_for_matko():
    commands = []
    for p in Path.cwd().glob('Chapter ?? --*.adoc'):
        # Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc
        toks = p.name.split()
        if len(toks) > 2:
            name = toks[0] + '-' + toks[1] + '_' + '-'.join([t for t in toks[2:] if t.strip('-')])
            name = name.replace('(', '').replace(')', '')
            cmd = f'git mv "{p.name}" "{name}"'
            print(cmd)
            if not input('Enter to skip'):
                continue
            commands.append(cmd)
    print('\n'.join(commands))
    with open('git-rename-commands.sh', 'wt') as fout:
        fout.writelines(commands)
    # more git - rename - commands.sh
    with open('git-rename-commands.sh', 'wt') as fout:
        fout.write('\n'.join(commands))
