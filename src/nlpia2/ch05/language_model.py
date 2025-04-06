from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
import re
import yaml

import spacy
nlp = spacy.load('en_core_web_md')


DATA_DIR = Path('../data').expanduser().resolve().absolute()


def generate_ngrams(seq):
    return list(zip([seq[:-1], seq[1:]]))


# def generate_ngrams(seq, n=2):
#     return list(zip([seq[i:-j] for i,j in zip(range(n-1), range(1,n))], seq[(n-1):]]))


def generate_sentence(probs=None):
    model_order = 0 if len(probs.shape) == 1 else 1
    # if model_order:
    #     model_order = len(probs.shape) - 1
    if model_order:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            ctemp = np.random.choice(probs.index.values, p=probs[c].values)
            if ctemp == '<SOS>':
                continue
            c = ctemp
            generated_sentence.append(c)
        return ''.join(generated_sentence)
    else:
        generated_sentence = []
        c = '<SOS>'
        while c != '<EOS>':
            c = np.random.choice(probs.index.values, p=probs.values)
            if c == '<SOS>':
                continue
            generated_sentence.append(c)
        return ''.join(generated_sentence)


def tokenize(s):
    return re.findall(r'[a-z ]', s.lower())


def load_token_lists(filepath=DATA_DIR / 'quotes.yml', tokenizer=tokenize):
    quotes = yaml.full_load(open(filepath))
    sentences = []
    for q in quotes:
        txt = q['text']
        doc = nlp(txt)
        sentences.extend([s.text for s in doc.sents])
    # convert to lists of chars
    sentences = [tokenizer(s) for s in sentences]
    sentences = [['<SOS>'] + clist + ['<EOS>'] for clist in sentences]
    return sentences


def calc_probs(token_lists):
    counters = [Counter(clist) for clist in sentences]
    df = pd.DataFrame(counters)
    # >>> df.sum()
    # <SOS>     174.0
    # t        1302.0
    # h         651.0
    # e        1635.0
    #          2674.0
    # g         289.0
    # ...
    # q           6.0
    # z          13.0
    # dtype: float64

    # df.head()
    #    <SOS>     t    h     e          g     i  ...    y    w    b   x   j   q   z
    # 0      1  10.0  5.0  12.0  17.0  5.0  11.0  ...  NaN  NaN  NaN NaN NaN NaN NaN
    # 1      1   4.0  4.0  14.0  22.0  4.0  10.0  ...  4.0  1.0  2.0 NaN NaN NaN NaN
    # 2      1   7.0  NaN  15.0  11.0  2.0  10.0  ...  1.0  NaN  1.0 NaN NaN NaN NaN
    # 3      1  10.0  5.0  13.0  17.0  2.0   8.0  ...  NaN  1.0  1.0 NaN NaN NaN NaN
    # 4      1   7.0  3.0  15.0  22.0  4.0   7.0  ...  1.0  NaN  3.0 NaN NaN NaN NaN
    # [5 rows x 29 columns]
    prob = df.sum()
    prob = prob / prob.sum()
    # prob
    # <SOS>    0.010677
    # t        0.079892
    # h        0.039946
    # e        0.100325
    #          0.164079
    # g        0.017733
    # ...
    return prob


def calc_conditional_probs(ngram_lists):
    """ FIXME """
    beginning_counts, ending_counts = Counter(), Counter()
    for ngrams in ngram_lists:
        beginnings, endings = zip(*ngrams)
        beginning_counts += Counter(beginnings)
        ending_counts += Counter(endings)
    df_probs = pd.DataFrame(np.zeros(len(endings), len(beginnings)), columns=beginnings.keys(), index=endings.keys())
    return df_probs.fillna(0)


if __name__ == '__main__':
    sentences = load_token_lists()
    sentence_2grams = list(generate_ngrams(s) for s in sentences)
    probs = calc_probs([[ngram for ngram[-1] in s] for s in sentence_2grams])
    generate_sentence(probs=probs)
    probs = calc_conditional_probs(sentence_2grams)
    generate_sentence()
    generate_sentence()
    generate_sentence()
