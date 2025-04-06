>>> ls -hal
>>> df = load_dataset()
>>> from collections import Counter
>>> from char_rnn_from_scratch_refactored import *
>>> df = load_dataset()
>>> groups = df.groupby('category')
>>> for i, g in groups:
...     print(i, g['text'].nunique() / len(g))
...
>>> g.columns
Index(['name', 'category'], dtype='object')
>>> for i, g in groups:
...     print(i, g['name'].nunique() / len(g))
...
>>> len(groups['Arabic'])
>>> groups
<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f9035ad09d0>
>>> g = df[df['category'] == 'Arabic']
>>> len(g)
2000
>>> g
         name category
0      Khoury   Arabic
1       Nahas   Arabic
2       Daher   Arabic
3      Gerges   Arabic
4      Nazari   Arabic
...       ...      ...
1995    Daher   Arabic
1996     Awad   Arabic
1997   Malouf   Arabic
1998  Mustafa   Arabic
1999    Aswad   Arabic

[2000 rows x 2 columns]
>>> g['name'].sort_values()
521     Abadi
289     Abadi
314     Abadi
918     Abadi
1146    Abadi
        ...  
921     Zogby
541     Zogby
981     Zogby
432     Zogby
548     Zogby
Name: name, Length: 2000, dtype: object
>>> hist -o -p
>>> hist -f char_rnn_name_nationality_nonrandom_sample.hist.py
>>> hist -o -p -f char_rnn_name_nationality_nonrandom_sample.hist.md
