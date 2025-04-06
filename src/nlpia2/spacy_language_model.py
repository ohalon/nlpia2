import spacy  # https://spacy.io

DEFAULT_MODEL_NAME = 'en_core_web_sm'
# dict of normalization "links" that must terminate in a cannonical name (leaf of the tree)
MODEL_NAMES = {
    # None or empty strings can be used to redirect to a default value/name 
    None: DEFAULT_MODEL_NAME,
    '': DEFAULT_MODEL_NAME,
    'default': DEFAULT_MODEL_NAME,
    # links to noncanonical names will be followed to until they are not found as keys
    'en': DEFAULT_MODEL_NAME,
    'english': DEFAULT_MODEL_NAME,

    'small': 'en_core_web_sm',
    'sm': 'en_core_web_sm',
    
    'medium': 'en_core_web_md',
    'md': 'en_core_web_md',
    
    'large': 'en_core_web_lg',
    'lg': 'en_core_web_lg',
    }


def cannonicalize_name(name=None, name_dict=MODEL_NAMES, max_redirects=5, default=None):
    """ Follow mappings to a normalized name using a dict of abbreviated_name: cannonical_name

    >>> cannonicalize_name('sm')
    'en_core_web_sm'
    >>> cannonicalize_name()
    'en_core_web_md'
    """
    for i in range(max_redirects):
        if name not in name_dict:
            if not i and default is not None:
                # first time checking for redirect abbreviation and no default specified
                return default
            # reached a leaf in the tree so name must be cannonical (a value in the kv dict) 
            return name
        name = name_dict[name]
    return name


def load(*args, **kwargs):
    """ Expand model_name abbreviations and download model weights before using spacy.load
    
    >>> nlp = load('en')
    >>> nlp.lang
    'en'
    >>> nlp.meta['name']
    'core_web_md'
    >>> load().meta['name']
    'core_web_md'
    """
    if 'name' not in kwargs and not len(args):
        args = [DEFAULT_MODEL_NAME]
    args = list(args)
    name = kwargs.pop('name', None) or args.pop(0)
    args = tuple([cannonicalize_name(name)] + list(args))
    try:
        nlp = spacy.load(*args, **kwargs)
    except OSError:
        spacy.cli.download(args[0])
        nlp = spacy.load(*args, **kwargs)
    return nlp


nlp = load()