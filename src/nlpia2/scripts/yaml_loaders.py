import yaml
import typing
import re
import subprocess
import sys
from collections.abc import Mapping

import logging
log = logging.getLogger(__name__)


class Loader(yaml.SafeLoader):
    pass


class Dumper(yaml.SafeDumper):
    pass


class Tagged(typing.NamedTuple):
    tag: str
    value: object


def construct_undefined(self, node):
    log.debug(node)
    if isinstance(node, yaml.nodes.ScalarNode):
        value = self.construct_scalar(node)
    elif isinstance(node, yaml.nodes.SequenceNode):
        value = self.construct_sequence(node)
    elif isinstance(node, yaml.nodes.MappingNode):
        value = self.construct_mapping(node)
        value['node_tag'] = value.get('node_tag', node.tag)
        return value
    else:
        assert False, f"unexpected node: {node!r}"
    for typ in (str, float, int, list, dict, bytes):
        if node.tag.lower().endswith(str(typ.__name__)):
            return typ(value)
        if isinstance(value, typ):
            return typ(value)
    return Tagged(node.tag, value)


Loader.add_constructor(None, construct_undefined)


def flatten(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = '{0}{1}{2}'.format(parent_key, sep, k) if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # apply itself to each element of the list - that's it!
            items.append((new_key, list(map(flatten, v))))
        else:
            items.append((new_key, v))
    return dict(items)


def get_blocks(doc):
    blocks = []
    for b in doc['blocks']:
        blocks.append((0, ' ' * 4 * 1, b['node_name']))
        for b1 in b['blocks']:
            blocks.append((1, ' ' * 4 * 2, b1['node_name']))
            for b2 in b1['blocks']:
                blocks.append((2, ' ' * 4 * 3, b2['node_name']))
                for b3 in b2['blocks']:  # there are no more 'section' blocks this deep (>2)
                    blocks.append((3, ' ' * 4 * 4, b3['node_name']))
                    for b4 in b3['blocks']:  # there are no more blocks this deep (>3)
                        blocks.append((4, ' ' * 4 * 5, b4['node_name']))
                        for b5 in b4['blocks']:
                            blocks.append((5, ' ' * 4 * 6, b4['node_name']))
    return blocks


if __name__ == '__main__':
    try:
        chapter_path = sys.argv[1]
    except IndexError:
        chapter_path = 'manuscript/adoc/Chapter 07 -- Getting Words in Order with Convolutional Neural Networks (CNNs).adoc'

    if '--flatten' in sys.argv:
        doflatten = True

    yaml_path = f'{chapter_path}.yml'
    # scripts/adoc2yaml.rb contains:
    #    yaml_output = File.open(chapter_path + '.yml', "w")
    #    yaml_output.write((Asciidoctor.load_file chapter_path).to_yaml)
    command = ['ruby', 'scripts/adoc2yaml.rb', chapter_path]
    stdoutput = subprocess.check_output(command)

    # re_reference = r'^\s*[-a-zA-Z0-9]+\:\ \*.*'
    # re_address = r'^-\ \&.*'

    lines = open(yaml_path).readlines()
    cleaned_lines = []
    for line in lines:
        cleaned = line
        # TODO: Should probably eliminate quote pointers to avoid recursively nested docs.
        match = re.match(r'^(\s*[-_a-zA-Z0-9]+\:\ )(\*1)$', cleaned)
        if match:
            cleaned = match.groups()[0] + '"' + match.groups()[1] + '"'
        # match = re.match(r'^(\s*[-_a-zA-Z0-9]+\:\ )\*(.*)', cleaned)
        # if match:
        #     cleaned = ''.join(match.groups())
        # match = re.match(r'^(\s*-)\ \&.*', cleaned)
        # if match:
        #     cleaned = ''.join(match.groups())
        cleaned_lines.append(cleaned)
    text = '\n'.join(cleaned_lines)
    doc = yaml.load(text, Loader=Loader)
    yaml.dump(doc, open(yaml_path, 'w'))
    # process the doc to extract/rearrange as you like
    print(yaml.dump(doc))
