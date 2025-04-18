""" DEPRECATED: Adds Hunspell pipe to spacy pipeline 

HunSpell requires OS-specific binary apps/data to be installed
"""

import sys
import logging

import numpy as np
import spacy

LANGS = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
LANGS_ABBREV = 'en enmd enlg'.split()
LANGS += 'de_core_news_sm de_core_news_md de_trf_bertbasecased_lg'.split()
LANGS_ABBREV += 'de demd delg'.split()
LANGS_ABBREV = dict(zip(LANGS_ABBREV, LANGS))

LANG = LANGS_ABBREV['en']


log = logging.getLogger('qary')
UNKNOWN_WORDVEC = np.array([])


class passthroughSpaCyPipe:
    """ Callable pass-through SpaCy Pipeline Component class (callable) for fallback if spacy_hunspell.spaCyHunSpell fails"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, doc):
        log.info(f"This passthroughSpaCyPipe component only logs the token count: {len(doc)}")
        return doc


try:
    import spacy_hunspell
except ImportError:
    log.debug('Failed to import spaCyHunSpell. Substituting with fake . . .')
    spacy_hunspell = None
spaCyHunSpell = spacy_hunspell.spaCyHunSpell if spacy_hunspell else passthroughSpaCyPipe


def add_hunspell_pipe(model):
    try:
        spacy.tokens.Token.get_extension('hunspell_spell')
        log.debug(f'SpaCy Token already has a hunspell Pipe section . . .')
        return model
    except ValueError:
        pass
    if sys.platform == 'linux' or sys.platform == 'linux2':
        hunspell = spaCyHunSpell(model, 'linux')
    elif sys.platform == 'darwin':
        hunspell = spaCyHunSpell(model, 'mac')
    else:  # sys.platform == 'win32':
        try:
            # TODO determine paths for en_US.dic and en_US.aff on windows
            hunspell = spaCyHunSpell(model, ('en_US.dic', 'en_US.aff'))
        except Exception:
            log.debug('Failed to locate en_US.dic and en_US.aff files. Substituting with fake . . .')
            hunspell = passthroughSpaCyPipe()
    try:
        model.add_pipe(hunspell)
    except ValueError:
        log.debug(f'SpaCy parser {model} already has a hunspell Pipe section...')
    return model


def load(lang=None):
    """ Load the specified language model or the small English model, if none specified

    >>> load()  # doctest: +ELLIPSIS
    <spacy.lang.en.English object at ...
    """
    global nlp, UNKNOWN_WORDVEC
    model = None
    log.info(f"Loading SpaCy model...")
    nlp_lang = getattr(nlp, 'lang', '')
    nlp_meta = getattr(nlp, 'meta', {})
    nlp_size = nlp_meta.get('name', '')[-2:]
    if nlp_lang and (not lang or (nlp_lang == lang[:2] and nlp_size == lang[-2:])):
        model = nlp
    if model is None and lang:
        try:
            model = spacy.load(lang)
        except OSError:
            spacy.cli.download(lang)
            model = spacy.load(lang)
    if model is None:
        if not lang:
            for lang in LANGS:
                try:
                    model = spacy.load(lang)
                    break
                except OSError:
                    pass
        else:
            if model is None:
                lang = LANG
                spacy.cli.download(lang)
                model = spacy.load(lang)
    log.info(
        f"Finished loading SpaCy model: {model} ({model.meta['lang']}_{model.meta['name']})\n"
        f"    with {model.meta['performance']['token_acc']:.2f}% token accuracy\n"
        f"     and {model.meta['performance']['ents_f']:.2f}% named entity recognition F1 score.\n"
    )
    model = add_hunspell_pipe(model)
    if nlp is None:
        nlp = model
    # load the highest accuracy model into the global singleton nlp variable (user can still override)
    if nlp.lang == 'en':
        if nlp.meta['performance']['token_acc'] < model.meta['performance']['token_acc']:
            nlp = model
    UNKNOWN_WORDVEC = np.random.randn(nlp._meta['vectors']['width'])
    UNKNOWN_WORDVEC /= np.linalg.norm(UNKNOWN_WORDVEC)
    return model  # return value may be lower accuracy, so `nlp=load('en_web_core_sm')` will have lower accuracy `nlp`


# FIXME: doesn't inherit from spacy.nlp so needs to be deleted in favor of load() function above
class SpacyLM:
    nlp = None
    lang = None

    def __init__(self, lang=None, eager=False):
        self.lang = lang
        if eager:
            self.nlp = load(self.lang)

    def __call__(self, *args, **kwargs):
        self.lang = kwargs.pop('lang', self.lang)
        if self.nlp is None:
            self.nlp = load(self.lang)
            # self.nlp.on_load()
        return self.nlp(*args)


def tokenize(s):
    global nlp
    return [tok.text for tok in nlp(s)]


if __name__ = '__main__':
    nlp = load(LANG)
