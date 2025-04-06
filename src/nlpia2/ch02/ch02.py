# ch02.ipy

"""
>>> text = ("Trust me, though, the words were on their way, and when "
...         "they arrived, Liesel would hold them in her hands like "
...         "the clouds, and she would wring them out, like the rain.")
>>> tokens = text.split()
>>> tokens[:9]
['Trust',
 'me,',
 'though,',
 'the',
 'words',
 'were',
 'on',
 'their',
 'way,']
"""

# NLPIA IDEA: sentence rewriting for data augmentation:
#       replace all re.replace('(,\ and )([c])', text, ['. \2'.upper(), '. And \2', '. In addition \2', '. Also ', '. Furthermore '])
#       obviously '\2' upper needs to be find_all() and manually replace
# NLPIA IDEA: generate multiple versions of a sentence and select the one that meets your criteria (brevity, clarity, reading score, style)


r"""
[source,python]
----
>>> import numpy as np  # <1>
>>> vocab = sorted(set(tokens))  # <2>
>>> ' '.join(vocab[:12])  # <3>
", . Survival There's adequate as fittest maybe most no of such"
>>> num_tokens = len(tokens)
>>> num_tokens
18
>>> vocab_size = len(vocab)
>>> vocab_size
15
----
<1> `str.split()` is your quick-and-dirty tokenizer.
<2> Coercing the `list` into a `set` ensures that your vocabulary contains only *unique* tokens that you want to keep track of.
<3> Sorted lexographically (lexically) so punctuation comes before letters, and capital letters come before lowercase letters.
"""
import numpy as np  # <1>
vocab = sorted(set(tokens))  # <2>
' '.join(vocab[:12])  # <3>
# ", . Survival There's adequate as fittest maybe most no of such"
num_tokens = len(tokens)
num_tokens
# 18
vocab_size = len(vocab)
vocab_size
# 15

r"""
>>> import re
>>> pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
>>> text = ("There's no such thing as survival of the fittest."
...         "Survival of the most adequate, maybe.")
>>> tokens = list(re.findall(pattern, text))

>>> tokens[:8]
["There's", 'no', 'such', 'thing', 'as', 'survival', 'of', 'the']

>>> tokens[8:16]
['fittest', '.', 'Survival', 'of', 'the', 'most', 'adequate', ',']

>>> tokens[16:]
['maybe', '.']
"""


r"""
>>> df_onehot = pd.DataFrame(onehot_vectors, columns=vocab)
>>> df_onehot.iloc[:,:8]
    ,  .  Survival  There's  adequate  as  fittest  maybe
0   0  0         0        1         0   0        0      0
1   0  0         0        0         0   0        0      0
2   0  0         0        0         0   0        0      0
3   0  0         0        0         0   0        0      0
4   0  0         0        0         0   1        0      0
5   0  0         0        0         0   0        0      0
6   0  0         0        0         0   0        0      0
7   0  0         0        0         0   0        0      0
8   0  0         0        0         0   0        1      0
9   0  1         0        0         0   0        0      0
10  0  0         1        0         0   0        0      0
11  0  0         0        0         0   0        0      0
12  0  0         0        0         0   0        0      0
13  0  0         0        0         0   0        0      0
14  0  0         0        0         1   0        0      0
15  1  0         0        0         0   0        0      0
16  0  0         0        0         0   0        0      1
17  0  1         0        0         0   0        0      0
"""


r"""
>>> import re
>>> pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
>>> text = ("There's no such thing as survival of the fittest."
...         "Survival of the most adequate, maybe.")
>>> tokens = list(re.findall(pattern, text))

>>> tokens[:8]
["There's", 'no', 'such', 'thing', 'as', 'survival', 'of', 'the']
>>> tokens[8:16]
['fittest', '.', 'Survival', 'of', 'the', 'most', 'adequate', ',']
>>> tokens[16:]
['maybe', '.']
"""
import re

r"""
>>> pattern = r'\w+(?:\'\w+)?|[^\w\s]'  # <1>
>>> text = ("There's no such thing as survival of the fittest."
            "Survival of the most adequate, maybe.")
>>> tokens = re.findall(pattern, text)
>>> tokens
["There's", 'no', 'such', 'thing', 'as', 'survival', ...]
"""


def tokenize(text, pattern=r'\w+(?:\'\w+)?|[^\w\s]'):
    r""" Split English text into words, ignoring only 1 internal punctuation"

    returns list(re.findall(r'\w+(?:\'\w+)?|[^\w\s]', text))
    """
    return list(re.findall(pattern, text))
