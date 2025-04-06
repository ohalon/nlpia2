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
>>> hist -o -p
>>> df
         name category
0      Khoury   Arabic
1       Nahas   Arabic
2       Daher   Arabic
3      Gerges   Arabic
4      Nazari   Arabic
...       ...      ...
20069    Zhai  Chinese
20070   Zhang  Chinese
20071     Zhi  Chinese
20072   Zhuan  Chinese
20073    Zhui  Chinese

[20074 rows x 2 columns]
>>> >>> fraction_unique = {}
... >>> for i, g in df.groupby('category'):
... ...     fraction_unique[i] = g['name'].nunique() / len(g)
... >>> pd.Series(fraction_unique).sort_values()
...
Arabic        0.054000
Chinese       0.917910
German        0.943370
Greek         0.950739
Dutch         0.962963
Czech         0.967245
Vietnamese    0.972603
Irish         0.974138
Spanish       0.983221
French        0.985560
Italian       0.988717
Polish        0.992806
Russian       0.992985
Japanese      0.998991
English       1.000000
Korean        1.000000
Portuguese    1.000000
Scottish      1.000000
dtype: float64
>>> from torch.nn import RNN
>>> htop
>>> !htop
>>> results = train()
>>> from collections import Counter
... >>> confusion = {c: Counter() for c in CATEGORIES}
... >>> counts = {}
... >>> for i, g in df.groupby('name'):
... ...      counts = Counter(g['category']) 
... ...      most_popular = sorted([(x[1], x[0]) for x in zip(counts.items())])[-1][1]
... ...      confusion[most_popular] += counts
...
>>> Counter(dict(a=2, b=3))
Counter({'a': 2, 'b': 3})
>>> from collections import Counter
... >>> confusion = {c: Counter() for c in CATEGORIES}
... >>> counts = {}
... >>> for i, g in df.groupby('name'):
... ...      counts = Counter(g['category']) 
... ...      most_popular = sorted([(x[1], x[0]) for x in counts.items()])[-1][1]
... ...      confusion[most_popular] += counts
...
>>> pd.DataFrame(confusion)
            Arabic  Irish  Spanish  French  German  English  Korean  ...  Greek  Czech  Italian  Portuguese  Russian  Dutch  Chinese
Arabic      2000.0    NaN      NaN     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN         NaN      NaN    NaN      NaN
English        3.0   54.0     25.0    34.0    35.0   3381.0     6.0  ...    1.0    NaN      2.0         3.0     20.0    NaN      NaN
Japanese       1.0    NaN      NaN     NaN     NaN      NaN     3.0  ...    NaN    NaN      NaN         NaN      1.0    NaN      NaN
German         1.0    3.0      6.0     NaN   692.0      NaN     2.0  ...    NaN    NaN      1.0         NaN     14.0    NaN      NaN
Irish          NaN  224.0      NaN     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN         NaN      2.0    NaN      NaN
French         NaN    2.0      7.0   250.0     9.0      NaN     NaN  ...    NaN    NaN      1.0         NaN      3.0    NaN      NaN
Dutch          NaN    1.0      2.0     3.0    24.0      8.0     NaN  ...    NaN    NaN      2.0         NaN      NaN  256.0      NaN
Spanish        NaN    NaN    297.0     NaN     NaN      NaN     NaN  ...    NaN    NaN      1.0         NaN      NaN    NaN      NaN
Russian        NaN    NaN      3.0     NaN     2.0      NaN     NaN  ...    NaN    NaN      NaN         NaN   9395.0    NaN      1.0
Italian        NaN    NaN     32.0     NaN     NaN      NaN     NaN  ...    NaN    NaN    674.0         1.0      NaN    NaN      NaN
Portuguese     NaN    NaN     38.0     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN        36.0      NaN    NaN      NaN
Polish         NaN    NaN      1.0     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN         NaN      3.0    NaN      NaN
Chinese        NaN    NaN      NaN     1.0     1.0      7.0    15.0  ...    NaN    NaN      1.0         NaN      4.0    1.0    227.0
Czech          NaN    NaN      NaN     1.0    14.0      7.0     NaN  ...    NaN  486.0      NaN         NaN      2.0    NaN      NaN
Korean         NaN    NaN      NaN     NaN     NaN      NaN    83.0  ...    NaN    NaN      NaN         NaN      3.0    NaN      1.0
Vietnamese     NaN    NaN      NaN     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN         NaN      NaN    NaN      NaN
Scottish       NaN    NaN      NaN     NaN     NaN      NaN     NaN  ...    NaN    NaN      NaN         NaN      NaN    NaN      NaN
Greek          NaN    NaN      NaN     NaN     NaN      NaN     NaN  ...  203.0    NaN      NaN         NaN      NaN    NaN      NaN

[18 rows x 18 columns]
>>> confusion = pd.DataFrame(confusion)
>>> confusion /= confusion.sum(axis=1)
>>> confusion
            Arabic     Irish   Spanish    French    German   English  ...     Czech   Italian  Portuguese   Russian     Dutch   Chinese
Arabic      1.0000       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN       NaN       NaN       NaN
English     0.0015  0.232759  0.083893  0.122744  0.048343  0.921756  ...       NaN  0.002821    0.040541  0.002126       NaN       NaN
Japanese    0.0005       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN  0.000106       NaN       NaN
German      0.0005  0.012931  0.020134       NaN  0.955801       NaN  ...       NaN  0.001410         NaN  0.001488       NaN       NaN
Irish          NaN  0.965517       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN  0.000213       NaN       NaN
French         NaN  0.008621  0.023490  0.902527  0.012431       NaN  ...       NaN  0.001410         NaN  0.000319       NaN       NaN
Dutch          NaN  0.004310  0.006711  0.010830  0.033149  0.002181  ...       NaN  0.002821         NaN       NaN  0.861953       NaN
Spanish        NaN       NaN  0.996644       NaN       NaN       NaN  ...       NaN  0.001410         NaN       NaN       NaN       NaN
Russian        NaN       NaN  0.010067       NaN  0.002762       NaN  ...       NaN       NaN         NaN  0.998618       NaN  0.003731
Italian        NaN       NaN  0.107383       NaN       NaN       NaN  ...       NaN  0.950635    0.013514       NaN       NaN       NaN
Portuguese     NaN       NaN  0.127517       NaN       NaN       NaN  ...       NaN       NaN    0.486486       NaN       NaN       NaN
Polish         NaN       NaN  0.003356       NaN       NaN       NaN  ...       NaN       NaN         NaN  0.000319       NaN       NaN
Chinese        NaN       NaN       NaN  0.003610  0.001381  0.001908  ...       NaN  0.001410         NaN  0.000425  0.003367  0.847015
Czech          NaN       NaN       NaN  0.003610  0.019337  0.001908  ...  0.936416       NaN         NaN  0.000213       NaN       NaN
Korean         NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN  0.000319       NaN  0.003731
Vietnamese     NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN       NaN       NaN       NaN
Scottish       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN       NaN       NaN       NaN
Greek          NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN         NaN       NaN       NaN       NaN

[18 rows x 18 columns]
>>> confusion.fillna(0, inplace=True)
>>> confusion
            Arabic     Irish   Spanish    French    German   English  ...     Czech   Italian  Portuguese   Russian     Dutch   Chinese
Arabic      1.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000000  0.000000  0.000000
English     0.0015  0.232759  0.083893  0.122744  0.048343  0.921756  ...  0.000000  0.002821    0.040541  0.002126  0.000000  0.000000
Japanese    0.0005  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000106  0.000000  0.000000
German      0.0005  0.012931  0.020134  0.000000  0.955801  0.000000  ...  0.000000  0.001410    0.000000  0.001488  0.000000  0.000000
Irish       0.0000  0.965517  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000213  0.000000  0.000000
French      0.0000  0.008621  0.023490  0.902527  0.012431  0.000000  ...  0.000000  0.001410    0.000000  0.000319  0.000000  0.000000
Dutch       0.0000  0.004310  0.006711  0.010830  0.033149  0.002181  ...  0.000000  0.002821    0.000000  0.000000  0.861953  0.000000
Spanish     0.0000  0.000000  0.996644  0.000000  0.000000  0.000000  ...  0.000000  0.001410    0.000000  0.000000  0.000000  0.000000
Russian     0.0000  0.000000  0.010067  0.000000  0.002762  0.000000  ...  0.000000  0.000000    0.000000  0.998618  0.000000  0.003731
Italian     0.0000  0.000000  0.107383  0.000000  0.000000  0.000000  ...  0.000000  0.950635    0.013514  0.000000  0.000000  0.000000
Portuguese  0.0000  0.000000  0.127517  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.486486  0.000000  0.000000  0.000000
Polish      0.0000  0.000000  0.003356  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000319  0.000000  0.000000
Chinese     0.0000  0.000000  0.000000  0.003610  0.001381  0.001908  ...  0.000000  0.001410    0.000000  0.000425  0.003367  0.847015
Czech       0.0000  0.000000  0.000000  0.003610  0.019337  0.001908  ...  0.936416  0.000000    0.000000  0.000213  0.000000  0.000000
Korean      0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000319  0.000000  0.003731
Vietnamese  0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000000  0.000000  0.000000
Scottish    0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000000  0.000000  0.000000
Greek       0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000    0.000000  0.000000  0.000000  0.000000

[18 rows x 18 columns]
>>> confusion.round(2)
            Arabic  Irish  Spanish  French  German  English  Korean  ...  Greek  Czech  Italian  Portuguese  Russian  Dutch  Chinese
Arabic         1.0   0.00     0.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
English        0.0   0.23     0.08    0.12    0.05     0.92    0.06  ...    0.0   0.00     0.00        0.04      0.0   0.00     0.00
Japanese       0.0   0.00     0.00    0.00    0.00     0.00    0.03  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
German         0.0   0.01     0.02    0.00    0.96     0.00    0.02  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Irish          0.0   0.97     0.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
French         0.0   0.01     0.02    0.90    0.01     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Dutch          0.0   0.00     0.01    0.01    0.03     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.86     0.00
Spanish        0.0   0.00     1.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Russian        0.0   0.00     0.01    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      1.0   0.00     0.00
Italian        0.0   0.00     0.11    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.95        0.01      0.0   0.00     0.00
Portuguese     0.0   0.00     0.13    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.49      0.0   0.00     0.00
Polish         0.0   0.00     0.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Chinese        0.0   0.00     0.00    0.00    0.00     0.00    0.16  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.85
Czech          0.0   0.00     0.00    0.00    0.02     0.00    0.00  ...    0.0   0.94     0.00        0.00      0.0   0.00     0.00
Korean         0.0   0.00     0.00    0.00    0.00     0.00    0.88  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Vietnamese     0.0   0.00     0.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Scottish       0.0   0.00     0.00    0.00    0.00     0.00    0.00  ...    0.0   0.00     0.00        0.00      0.0   0.00     0.00
Greek          0.0   0.00     0.00    0.00    0.00     0.00    0.00  ...    1.0   0.00     0.00        0.00      0.0   0.00     0.00

[18 rows x 18 columns]
>>> hist -o -p -f char_rnn_name_nationality_confusion.hist.md
>>> confusion = confusion[confusion.index]
>>> confusion
            Arabic   English  Japanese    German     Irish    French  ...   Chinese     Czech    Korean  Vietnamese  Scottish     Greek
Arabic      1.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
English     0.0015  0.921756  0.000000  0.048343  0.232759  0.122744  ...  0.000000  0.000000  0.063830    0.027397      0.99  0.004926
Japanese    0.0005  0.000000  0.994955  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.031915    0.000000      0.00  0.000000
German      0.0005  0.000000  0.000000  0.955801  0.012931  0.000000  ...  0.000000  0.000000  0.021277    0.013699      0.01  0.000000
Irish       0.0000  0.000000  0.000000  0.000000  0.965517  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.06  0.000000
French      0.0000  0.000000  0.000000  0.012431  0.008621  0.902527  ...  0.000000  0.000000  0.000000    0.000000      0.01  0.000000
Dutch       0.0000  0.002181  0.001009  0.033149  0.004310  0.010830  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
Spanish     0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
Russian     0.0000  0.000000  0.000000  0.002762  0.000000  0.000000  ...  0.003731  0.000000  0.000000    0.041096      0.04  0.000000
Italian     0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
Portuguese  0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
Polish      0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  0.000000
Chinese     0.0000  0.001908  0.002018  0.001381  0.000000  0.003610  ...  0.847015  0.000000  0.159574    0.109589      0.01  0.000000
Czech       0.0000  0.001908  0.001009  0.019337  0.000000  0.003610  ...  0.000000  0.936416  0.000000    0.000000      0.03  0.000000
Korean      0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.003731  0.000000  0.882979    0.095890      0.00  0.000000
Vietnamese  0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    1.000000      0.00  0.000000
Scottish    0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      1.00  0.000000
Greek       0.0000  0.000000  0.000000  0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000    0.000000      0.00  1.000000

[18 rows x 18 columns]
>>> confusion.round(2)
            Arabic  English  Japanese  German  Irish  French  Dutch  ...  Polish  Chinese  Czech  Korean  Vietnamese  Scottish  Greek
Arabic         1.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      0.00    0.0
English        0.0     0.92      0.00    0.05   0.23    0.12   0.00  ...    0.02     0.00   0.00    0.06        0.03      0.99    0.0
Japanese       0.0     0.00      0.99    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.03        0.00      0.00    0.0
German         0.0     0.00      0.00    0.96   0.01    0.00   0.00  ...    0.02     0.00   0.00    0.02        0.01      0.01    0.0
Irish          0.0     0.00      0.00    0.00   0.97    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      0.06    0.0
French         0.0     0.00      0.00    0.01   0.01    0.90   0.00  ...    0.03     0.00   0.00    0.00        0.00      0.01    0.0
Dutch          0.0     0.00      0.00    0.03   0.00    0.01   0.86  ...    0.00     0.00   0.00    0.00        0.00      0.00    0.0
Spanish        0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      0.00    0.0
Russian        0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.04      0.04    0.0
Italian        0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.01     0.00   0.00    0.00        0.00      0.00    0.0
Portuguese     0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      0.00    0.0
Polish         0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.97     0.00   0.00    0.00        0.00      0.00    0.0
Chinese        0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.85   0.00    0.16        0.11      0.01    0.0
Czech          0.0     0.00      0.00    0.02   0.00    0.00   0.00  ...    0.04     0.00   0.94    0.00        0.00      0.03    0.0
Korean         0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.88        0.10      0.00    0.0
Vietnamese     0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        1.00      0.00    0.0
Scottish       0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      1.00    0.0
Greek          0.0     0.00      0.00    0.00   0.00    0.00   0.00  ...    0.00     0.00   0.00    0.00        0.00      0.00    1.0

[18 rows x 18 columns]
>>> confusion.plot?
>>> import seaborn
>>> cax = plt.matshow(confusion)
... fig.colorbar(cax)
... 
... # Set up axes
... ax.set_xticklabels([''] + all_categories, rotation=90)
... ax.set_yticklabels([''] + all_categories)
... 
... # Force label at every tick
... ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
... ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
... 
... # sphinx_gallery_thumbnail_number = 2
... plt.show()
...
>>> fig = figure()
>>> fig = plt.figure()
>>> cax = plt.matshow(confusion)
... fig.colorbar(cax)
... 
... # Set up axes
... ax.set_xticklabels([''] + all_categories, rotation=90)
... ax.set_yticklabels([''] + all_categories)
... 
... # Force label at every tick
... ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
... ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
... 
... # sphinx_gallery_thumbnail_number = 2
... plt.show()
...
>>> ax = fig.add_subplot(111)
>>> cax = plt.matshow(confusion)
... fig.colorbar(cax)
... 
... # Set up axes
... ax.set_xticklabels([''] + all_categories, rotation=90)
... ax.set_yticklabels([''] + all_categories)
... 
... # Force label at every tick
... ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
... ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
... 
... # sphinx_gallery_thumbnail_number = 2
... plt.show()
...
>>> fig = plt.figure()
... ax = fig.add_subplot(111)
... cax = ax.matshow(confusion)
... fig.colorbar(cax)
... 
... # Set up axes
... ax.set_xticklabels([''] + confusion.columns, rotation=90)
... ax.set_yticklabels([''] + confusion.columns)
... 
... # Force label at every tick
... ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
... ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
... 
... # sphinx_gallery_thumbnail_number = 2
... plt.show()
...
>>> hist
>>> seaborn.set_style()
>>> seaborn.set_theme()
>>> hist -o -p -f char_rnn_name_nationality_dataset_confusion.hist.md
