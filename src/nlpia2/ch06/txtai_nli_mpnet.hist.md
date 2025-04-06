>>> import pandas as pd
>>> import textai as tai
>>> import txtai as tai
>>> import pandas as pd
>>> df = pd.read_csv('data/nlpia_lines.csv')
>>> df
       Unnamed: 0                                          line_text  line_number  ... is_comment  num_types  is_type_defined
0               0  = Natural Language Processing in Action, Secon...            0  ...      False          1             True
1               1                                        :chapter: 1            1  ...      False          1             True
2               2                                           :part: 1            2  ...      False          1             True
3               3                                         :sectnums:            3  ...      False          1             True
4               4                                      :imagesdir: .            4  ...      False          1             True
...           ...                                                ...          ...  ...        ...        ...              ...
14373       14373  * Efficiently modeling natural language charac...         1769  ...      False          1             True
14374       14374  * Weights in an RNN are adjusted in aggregate ...         1770  ...      False          1             True
14375       14375  * You can use different methods to examine the...         1771  ...      False          1             True
14376       14376  * You can model the natural language sequence ...         1772  ...      False          1             True
14377       14377                                                NaN         1773  ...      False          1             True

[14378 rows x 16 columns]
>>> df = pd.read_csv('data/nlpia_lines.csv', index_col=0)
>>> df
                                               line_text  line_number  ... num_types  is_type_defined
0      = Natural Language Processing in Action, Secon...            0  ...         1             True
1                                            :chapter: 1            1  ...         1             True
2                                               :part: 1            2  ...         1             True
3                                             :sectnums:            3  ...         1             True
4                                          :imagesdir: .            4  ...         1             True
...                                                  ...          ...  ...       ...              ...
14373  * Efficiently modeling natural language charac...         1769  ...         1             True
14374  * Weights in an RNN are adjusted in aggregate ...         1770  ...         1             True
14375  * You can use different methods to examine the...         1771  ...         1             True
14376  * You can model the natural language sequence ...         1772  ...         1             True
14377                                                NaN         1773  ...         1             True

[14378 rows x 15 columns]
>>> df.columns
Index(['line_text', 'line_number', 'filename', 'is_text', 'is_empty',
       'is_code_or_output', 'is_title', 'is_metadata', 'is_code_comment',
       'is_markup', 'is_figure_name', 'is_separator', 'is_comment',
       'num_types', 'is_type_defined'],
      dtype='object')
>>> df[df['is_code_or_output']]
                                               line_text  line_number  ... num_types  is_type_defined
129    image::../images/ch01/text-NLU-vector-graphviz...          129  ...         1             True
167    image::../images/ch01/vector-NLG-text-graphviz...          167  ...         1             True
441    image::../images/ch01/nlp-applications.png['Ne...          441  ...         1             True
524                                                 ----          524  ...         2             True
527    async function isPositive(text: string): Promi...          527  ...         1             True
...                                                  ...          ...  ...       ...              ...
14330                                               ----         1726  ...         2             True
14331              >>> print(' '.join(w for w in words))         1727  ...         1             True
14332                                                ...         1728  ...         2             True
14334                                                ...         1730  ...         2             True
14335                                               ----         1731  ...         2             True

[3479 rows x 15 columns]
>>> df[~df['line_number']]['filename']
>>> ~df['line_number']
0          -1
1          -2
2          -3
3          -4
4          -5
         ... 
14373   -1770
14374   -1771
14375   -1772
14376   -1773
14377   -1774
Name: line_number, Length: 14378, dtype: int64
>>> df['filename'][df['line_number'] == 1]
1        Chapter-01_Machines-that-can-read-and-write-NL...
1200     Chapter-02_Tokens-of-thought-natural-language-...
3579        Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc
5092     Chapter-04_Finding-Meaning-in-Word-Counts-Sema...
6857     Chapter-05_Word-Brain-artificial-neural-networ...
8339     Chapter-06_Reasoning-with-word-embeddings-word...
10647    Chapter-07_Finding-Kernels-of-Knowledge-in-Tex...
12605    Chapter-08_Reduce,-Reuse,-Recycle-Recurrent-Ne...
Name: filename, dtype: object
>>> from txtai.embeddings import Embeddings
... 
... # Create embeddings model, backed by sentence-transformers & transformers
... embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
...
>>> hist -f txtai_nli_mpnet.py
>>> hist -o -p -f txtai_nli_mpnet.hist.md
