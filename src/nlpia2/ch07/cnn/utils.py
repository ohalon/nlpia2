import sys


def parse_argv(sys_argv=sys.argv):
    argv = list(reversed(sys_argv[1:]))

    pipeline_args = []
    pipeline_kwargs = {}  # dict(tokenizer='tokenize_re')
    while len(argv):
        a = argv.pop()
        if a.startswith('--'):
            if '=' in a:
                k, v = a.split('=')
                k = k.lstrip('-')
            else:
                k = a.lstrip('-')
                v = argv.pop()
            pipeline_kwargs[k] = v
        else:
            pipeline_args.append(a)

    return pipeline_args, pipeline_kwargs
