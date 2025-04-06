import os
import sys
import subprocess
from pathlib import Path
import json
import logging

import xmltodict
from xml.parsers.expat import ExpatError


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
print(HOME_CODE_DIR)
assert HOME_CODE_DIR.name == 'code'
from nlpia2.constants import MANUSCRIPT_DIR
assert MANUSCRIPT_DIR.is_dir()
IMAGE_DIR = MANUSCRIPT_DIR / 'images'
assert IMAGE_DIR.is_dir()
SCRIPT_WORKING_DIR = os.getcwd()


def render_adoc(doctype='book', backend='html5', destination_dir='html', embedded=False, adoc_path='adoc/*.adoc'):
    """ Render adoc files in manuscript/asc to HTML or other viewable/printable format

    Input:
      backend (str): html5 xhtml5 docbook5 manpage
      doctype (str): article book manpage inline
      embedded (bool): whether to suppress enclosing document structure
    """

    # exit_code = subprocess.call(cmd, shell=True)  # exit_code == 0 if successful

    command = f'asciidoctor -d {doctype} -b {backend} -D {destination_dir} {adoc_path}'.split()
    return run(command=command, chdir=MANUSCRIPT_DIR)


def html2xml(destination_dir=None, html_path=None):
    """ Render adoc files in manuscript/asc to HTML or other viewable/printable format

    Implements `pandoc -t json -o ../xml/"$p".xml --indent=2 "$p"` in python.
    """
    # exit_code = subprocess.call(cmd, shell=True)  # exit_code == 0 if successful
    if html_path:
        htmlpaths = list(Path(html_path).glob(html_path))
    else:
        htmlpaths = list((MANUSCRIPT_DIR / 'xhtml5').glob('*.html'))

    if not destination_dir:
        destination_dir = MANUSCRIPT_DIR / 'xml'
    destination_dir.mkdir(exist_ok=True)

    command_messages = []
    for p in htmlpaths:
        # pandoc -t xml -o ../xml/"$p".xml 
        command = f'pandoc -t xml -o {str(destination_dir)} {p}'.split()
        output = run(command=command)
        messages.append(' '.join(command), output)
    return command_messages


def run(command, chdir=None):
    if chdir:
        log.info(f'Temporarily changing working directory to {chdir}')
        initial_cwd = os.getcwd()
        os.chdir(chdir)

    log.warning(f'Running: {" ".join(command)}')
    output = subprocess.run(command, capture_output=True)

    if chdir:
        os.chdir(initial_cwd)

    return {
        'stderr': output.stderr.decode("utf-8").splitlines(),
        'stdout': output.stderr.decode("utf-8").splitlines()
    }


def svg2png(filepath, dpi=300, width="100%", height="100%", background_color="white"):
    # exit_code = subprocess.call(cmd, shell=True)  # exit_code == 0 if successful
    filepath_noext = '.'.join(filepath.split('.')[:-1])
    # deprecated: cmd = f'inkscape --without-gui {filepath_noext}.svg -o {filepath_noext}.png'.split()
    cmd = str.split(f' inkscape {filepath_noext}.svg'
                    f' --export-filename {filepath_noext}.png'
                    f' --export-background={background_color}'
                    f' --export-dpi={dpi}'
                    # f'--export-width={width} --export-height={height}'
                    )
    return run(command=cmd, chdir=None)


def pprint_output(output, command=None):
    if command:
        print(f'\n{command} OUTPUT:')
    try:
        if output.get('stderr') or output.get('stdout'):
            json.dumps(str([output]), indent=4)
    except AttributeError:
        for cmd, outp in output:
            pprint_output(output=outp, command=cmd)


if __name__ == '__main__':
    
    # svgfilepaths = list(IMAGE_DIR.glob('**/*.svg'))
    default_svgfilepaths = [
        IMAGE_DIR / 'ch02' / 'survival-of-adequate-sentence-diagram.svg',
        ]
    default_svgfilepaths = [str(p) for p in default_svgfilepaths]    

    # FIXME: argparse!!
    if len(sys.argv) >= 2:
        args = sys.argv[1:]
    else:
        args = []
    flags = {'x': False, 'j': False, 'h': False, 's': False}
    for flag in '-x -j -h --xml --json --html'.split():
        if flag in args:
            f = flag.lstrip('-')[:1]
            flags[f] = True
            del args[args.index(flag)]
    # make sure dependencies are rendered
    if flags['j']:
        flags['x'] = flags['x'] or True
    if flags['h'] or flags['x']:
        flags['s'] = flags['s'] or True
    log.warning(f'flags = {flags}')

    if flags['s']:
        for filepath in (IMAGE_DIR).glob('**/*.svg'):
            filepath = str(filepath)
            output = svg2png(filepath=filepath)
            pprint_output(output, command=f'svg2png({filepath}')
    
    if flags['h']:
        output = render_adoc(
        doctype='book',
        backend='html5',
        destination_dir='html', 
        embedded=False)
        pprint_output(output, command='render_adoc(book, backend=html5, destination_dir=html, embedded=False)')

    # XML required for json
    if flags['x']:
        print('render_html(backend=docbook) STDOUT:')
        output = render_adoc(
            doctype='book', 
            backend='docbook', 
            destination_dir='xml', 
            embedded=False
            )
        pprint_output(output, command='render_adoc(book, backend=docbook, destination_dir=xml, embedded=False)')

    if flags['j']:
        bookdict = {}
        for filepath in (MANUSCRIPT_DIR / 'xml').glob('*.xml'):
            log.debug(f'Parsing {filepath}...')

            xml_text = open(filepath).read()
            d = None
            try:
                d = xmltodict.parse(xml_text)
            except ExpatError as e:
                log.error(f'Unable to parse xml in {filepath}:\n{e}')
                continue
            basename = filepath.with_suffix('').name
            bookdict[basename] = d
            filepath = MANUSCRIPT_DIR / 'json' / (basename + '.json')
            fout = open(filepath, 'wt')
            json.dump(d, fout, indent=4)
        filepath = MANUSCRIPT_DIR / 'json' / 'book.json'
        json.dump(d, fout, indent=4)
