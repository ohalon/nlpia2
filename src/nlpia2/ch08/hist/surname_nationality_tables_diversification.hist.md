# Scrape names for underrepresented countries

A lot of cutting edge data science is happening in Nigeria, Etheopia and other underrepresented countries.
We've included a few names from our multilingual chatbots and some of our intern rosters in the US and Africa.
The code here shows how to augment the PyTorch data with scraped names from `forebears.io`.
The `char_rnn_from_scratch_refactored` module and script has been moved and renamed here: [src/nlpia2/ch08/rnn_char/ch08_rnn_char_nationality.py](https://gitlab.com/tangibleai/nlpia2/-/tree/main/src/nlpia2/ch08/rnn_char/ch08_rnn_char_nationality.py)


```python
>>> %run char_rnn_from_scratch_refactored
>>> df = load_names_from_text(dedupe=True, categories=None)
>>> len(df.sample(100))
100
>>> len(df.groupby('category').sample(100))
>>> len(df.groupby('category').sample(100, replace=True))
2200
>>> df.sample?
>>> len(df.groupby('category').sample(frac=1.0, replace=True))
19241
>>> len(df)
19241
>>> categories
>>> CATEGORIES
['Arabic',
 'Irish',
 'Spanish',
 'French',
 'German',
 'English',
 'Korean',
 'Vietnamese',
 'Scottish',
 'Japanese',
 'Polish',
 'Greek',
 'Czech',
 'Italian',
 'Portuguese',
 'Russian',
 'Dutch',
 'Chinese',
 'Indian',
 'Ethiopian',
 'Nigerian',
 'Nepalese']
>>> CATEGORIES == df['category'].unique()
array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False])
>>> [c in CATEGORIES for c in df['category'].unique()]
[True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True,
 True]
>>> len(CATEGORIES)
22
>>> pd.read_html('https://forebears.io/ethiopia/surnames')
>>> s = """
... Rank    Surname Incidence   Frequency
... 1   Tesfaye 1,167,260   1:84
... 2   Mohammed    1,084,839   1:90
... 3   Getachew    895,366 1:109
... 4   Abebe   825,501 1:118
... 5   Girma   822,765 1:119
... 6   Tadesse 816,808 1:119
... 7   Solomon 672,249 1:145
... 8   Kebede  660,498 1:148
... 9   Bekele  635,868 1:153
... 
... 321 Shemsu  54,250  1:1,798
... 322 Lakew   54,089  1:1,803
... 323 Yoseph  53,767  1:1,814
... 324 Gebremariam 53,606  1:1,820
... 325 Sileshi 53,445  1:1,825
... 326 Degu    53,284  1:1,831
... 327 Zegeye  52,962  1:1,842
... 328 Halima  52,801  1:1,847
... """
...
>>> !curl -O https://forebears.io/ethiopia/surnames
>>> more surnames
>>> pd.read_html('/home/hobs/Downloads/surnames/Most Common Ethiopian Surnames & Meanings.html')
[     Rank      Surname  Incidence Frequency
 0       1      Tesfaye    1167260      1:84
 1       2     Mohammed    1084839      1:90
 2       3     Getachew     895366     1:109
 3       4        Abebe     825501     1:118
 4       5        Girma     822765     1:119
 ..    ...          ...        ...       ...
 323   324  Gebremariam      53606   1:1,820
 324   325      Sileshi      53445   1:1,825
 325   326         Degu      53284   1:1,831
 326   327       Zegeye      52962   1:1,842
 327   328       Halima      52801   1:1,847
 
 [328 rows x 4 columns]]
>>> surn = [] ; surn.append(pd.read_html('/home/hobs/Downloads/surnames/Most Common Ethiopian Surnames & Meanings.html')[0])
>>> surn = [] ; surn.append(pd.read_html('/home/hobs/Downloads/surnames/Most Common Malaysian Surnames & Meanings.html')[0])
>>> surn = [];
>>> for name in 'Malaysian Ethiopian Nigerian'.split():
...     sn = pd.read_html(f"/home/hobs/Downloads/surnames/Most Common {name} Surnames & Meanings.html")[0]
...     sn['category'] = name
...     surn.append(sn)
...
>>> urls = ['https://forebears.io/papua-new-guinea/surnames', 'https://forebears.io/malaysia/surnames', 'https://forebears.io/nigeria/surnames'] ; urls2 = ['https://forebears.io/papua-new-guinea#surnames']
>>> who
>>> from pathlib import Path
>>> Path.home / 'hobs'
>>> Path.home() / 'hobs' / 'Downloads' / 'surnames'
PosixPath('/home/hobs/hobs/Downloads/surnames')
>>> data_dir = _
>>> filepaths = data_dir.glob('Most Common *.html'))
>>> filepaths = list(data_dir.glob('Most Common *.html'))
>>> filepaths
[]
>>> data_dir.isfile()
>>> data_dir.is_file()
False
>>> data_dir.is_dir()
False
>>> data_dir
PosixPath('/home/hobs/hobs/Downloads/surnames')
>>> data_dir = Path.home() / 'Downloads' / 'surnames'
>>> filepaths = list(data_dir.glob('Most Common *.html'))
>>> filepaths
[PosixPath('/home/hobs/Downloads/surnames/Most Common Ethiopian Surnames & Meanings.html'),
 PosixPath('/home/hobs/Downloads/surnames/Most Common Nigerian Surnames & Meanings.html'),
 PosixPath('/home/hobs/Downloads/surnames/Most Common Malaysian Surnames & Meanings.html'),
 PosixPath('/home/hobs/Downloads/surnames/Most Common Papua New Guinean Surnames & Meanings.html')]
>>> dfs = []
... for fp in filepaths:
...     dfs.extend(pd.read_html(fp))
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     dfs.extend(pd.read_html(str(fp)))
... df = pd.concat(dfs)
...
>>> df
     Rank   Surname  Incidence Frequency
0       1   Tesfaye    1167260      1:84
1       2  Mohammed    1084839      1:90
2       3  Getachew     895366     1:109
3       4     Abebe     825501     1:118
4       5     Girma     822765     1:119
..    ...       ...        ...       ...
358   359      Kawi       1414   1:5,766
359   360      Koko       1412   1:5,775
360   361     Dokta       1409   1:5,787
361   362       Gau       1407   1:5,795
362   363      Karu       1401   1:5,820

[2692 rows x 4 columns]
>>> len(df)
2692
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('& Meanings', '').strip()
...     fp = str(fp)
...     
...     dfs.extend(pd.read_html(str(fp)))
...     dfs[-1]['country'] = country
... df = pd.concat(dfs)
...
>>> df
     Rank   Surname  Incidence Frequency                     country
0       1   Tesfaye    1167260      1:84          Ethiopian Surnames
1       2  Mohammed    1084839      1:90          Ethiopian Surnames
2       3  Getachew     895366     1:109          Ethiopian Surnames
3       4     Abebe     825501     1:118          Ethiopian Surnames
4       5     Girma     822765     1:119          Ethiopian Surnames
..    ...       ...        ...       ...                         ...
358   359      Kawi       1414   1:5,766  Papua New Guinean Surnames
359   360      Koko       1412   1:5,775  Papua New Guinean Surnames
360   361     Dokta       1409   1:5,787  Papua New Guinean Surnames
361   362       Gau       1407   1:5,795  Papua New Guinean Surnames
362   363      Karu       1401   1:5,820  Papua New Guinean Surnames

[2692 rows x 5 columns]
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     
...     dfs.extend(pd.read_html(str(fp)))
...     dfs[-1]['country'] = country
... df = pd.concat(dfs)
...
>>> df
     Rank   Surname  Incidence Frequency            country
0       1   Tesfaye    1167260      1:84          Ethiopian
1       2  Mohammed    1084839      1:90          Ethiopian
2       3  Getachew     895366     1:109          Ethiopian
3       4     Abebe     825501     1:118          Ethiopian
4       5     Girma     822765     1:119          Ethiopian
..    ...       ...        ...       ...                ...
358   359      Kawi       1414   1:5,766  Papua New Guinean
359   360      Koko       1412   1:5,775  Papua New Guinean
360   361     Dokta       1409   1:5,787  Papua New Guinean
361   362       Gau       1407   1:5,795  Papua New Guinean
362   363      Karu       1401   1:5,820  Papua New Guinean

[2692 rows x 5 columns]
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank')
...     df.columns = 'surname count frequency'.split()
...     df['country'] = country
...     df.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank', axis=1)
...     df.columns = 'surname count frequency'.split()
...     df['country'] = country
...     df.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank', axis=0)
...     df.columns = 'surname count frequency'.split()
...     df['country'] = country
...     df.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     # df = df.drop('Rank', axis=0)
...     df.columns = 'rank surname count frequency'.split()
...     df['country'] = country
...     df.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank', axis=1)
...     df.columns = 'rank surname count frequency'.split()
...     df['country'] = country
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank', axis=1)
...     df.columns = 'surname count frequency'.split()
...     df['country'] = country
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> df
      surname    count frequency            country
0     Tesfaye  1167260      1:84          Ethiopian
1    Mohammed  1084839      1:90          Ethiopian
2    Getachew   895366     1:109          Ethiopian
3       Abebe   825501     1:118          Ethiopian
4       Girma   822765     1:119          Ethiopian
..        ...      ...       ...                ...
358      Kawi     1414   1:5,766  Papua New Guinean
359      Koko     1412   1:5,775  Papua New Guinean
360     Dokta     1409   1:5,787  Papua New Guinean
361       Gau     1407   1:5,795  Papua New Guinean
362      Karu     1401   1:5,820  Papua New Guinean

[2692 rows x 4 columns]
>>> dfs = []
... for fp in filepaths:
...     nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df = df.drop('Rank', axis=1)
...     df.columns = 'surname count frequency'.split()
...     df['nationality'] = nationality
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> hist
>>> hist -o -p -f surname_nationality_tables_diversification.hist.md
>>> hist -f surname_nationality_tables_diversification.hist.py
>>> diversity_df = df.copy()
>>> %run char_rnn_from_scratch_refactored
>>> df = load_names_from_text(dedupe=True, categories=None)
>>> df_diversity.columns
>>> diversity_df.columns
Index(['surname', 'count', 'frequency', 'nationality'], dtype='object')
>>> df.columns
Index(['name', 'category', 'count'], dtype='object')
>>> df.columns = 'surname nationality count'.split()
>>> dftot = pd.concat([df, diversity_df])
>>> len(dftot)
21933
>>> dftot.shape
(21933, 4)
>>> diversity_df.shape
(2692, 4)
>>> df.shape
(19241, 3)
>>> dftot.sample(100)
         surname nationality  count frequency
4017     Noggins     English      1       NaN
8371     Neisser       Czech      1       NaN
621          Mak   Malaysian   6818   1:4,326
9522   Abramenko     Russian      1       NaN
7262     Miyoshi    Japanese      1       NaN
...          ...         ...    ...       ...
7509   Shinozuka    Japanese      1       NaN
11042   Bekrenev     Russian      1       NaN
16931      Tropp     Russian      1       NaN
2475      Eilers     English      1       NaN
7309     Nakadan    Japanese      1       NaN

[100 rows x 4 columns]
>>> df = diversity_df.copy()
>>> df.groupby('nationality')['count'].sum()
nationality
Ethiopian             52385528
Malaysian             18432096
Nigerian             117683728
Papua New Guinean      2448643
Name: count, dtype: int64
>>> groups = df.groupby('nationality')
>>> dfs
[         surname    count frequency nationality
 0        Tesfaye  1167260      1:84   Ethiopian
 1       Mohammed  1084839      1:90   Ethiopian
 2       Getachew   895366     1:109   Ethiopian
 3          Abebe   825501     1:118   Ethiopian
 4          Girma   822765     1:119   Ethiopian
 ..           ...      ...       ...         ...
 323  Gebremariam    53606   1:1,820   Ethiopian
 324      Sileshi    53445   1:1,825   Ethiopian
 325         Degu    53284   1:1,831   Ethiopian
 326       Zegeye    52962   1:1,842   Ethiopian
 327       Halima    52801   1:1,847   Ethiopian
 
 [328 rows x 4 columns],
        surname    count frequency nationality
 0      Ibrahim  3310419      1:54    Nigerian
 1         Musa  3039701      1:58    Nigerian
 2     Abubakar  2800579      1:63    Nigerian
 3    Abdullahi  2553566      1:69    Nigerian
 4     Mohammed  2338925      1:76    Nigerian
 ..         ...      ...       ...         ...
 995     Anyawu    19765   1:8,962    Nigerian
 996      Nneji    19754   1:8,967    Nigerian
 997     Ugwoke    19749   1:8,970    Nigerian
 998   Kingsley    19727   1:8,980    Nigerian
 999        Ama    19694   1:8,995    Nigerian
 
 [1000 rows x 4 columns],
      surname   count frequency nationality
 0        Tan  404514      1:73   Malaysian
 1        Lim  340271      1:87   Malaysian
 2        Lee  338534      1:87   Malaysian
 3       Wong  288771     1:102   Malaysian
 4        Wan  235819     1:125   Malaysian
 ...      ...     ...       ...         ...
 996      Leo    3458   1:8,529   Malaysian
 997    Haris    3457   1:8,532   Malaysian
 998    Balan    3455   1:8,537   Malaysian
 999     Hock    3453   1:8,542   Malaysian
 1000   Siang    3453   1:8,542   Malaysian
 
 [1001 rows x 4 columns],
     surname   count frequency        nationality
 0      John  238519      1:34  Papua New Guinean
 1     Peter  181519      1:45  Papua New Guinean
 2      Paul  120946      1:67  Papua New Guinean
 3     David  108827      1:75  Papua New Guinean
 4     James   98569      1:83  Papua New Guinean
 ..      ...     ...       ...                ...
 358    Kawi    1414   1:5,766  Papua New Guinean
 359    Koko    1412   1:5,775  Papua New Guinean
 360   Dokta    1409   1:5,787  Papua New Guinean
 361     Gau    1407   1:5,795  Papua New Guinean
 362    Karu    1401   1:5,820  Papua New Guinean
 
 [363 rows x 4 columns]]
>>> len(dfs)
4
>>> df = dfs[-1]
>>> df['normalized_count'] = df['count'] / df['count'].sum()
>>> df
    surname   count frequency        nationality  normalized_count
0      John  238519      1:34  Papua New Guinean          0.097409
1     Peter  181519      1:45  Papua New Guinean          0.074130
2      Paul  120946      1:67  Papua New Guinean          0.049393
3     David  108827      1:75  Papua New Guinean          0.044444
4     James   98569      1:83  Papua New Guinean          0.040255
..      ...     ...       ...                ...               ...
358    Kawi    1414   1:5,766  Papua New Guinean          0.000577
359    Koko    1412   1:5,775  Papua New Guinean          0.000577
360   Dokta    1409   1:5,787  Papua New Guinean          0.000575
361     Gau    1407   1:5,795  Papua New Guinean          0.000575
362    Karu    1401   1:5,820  Papua New Guinean          0.000572

[363 rows x 5 columns]
>>> df['normalized_freq'] = df['frequency'].str.split(':')[0].astype(float) / df['frequency'].str.split(':')[1].astype(float)
>>> df['normalized_freq'] = df['frequency'].str.split(':').apply(lambda x: float(x[0])) / df['frequency'].str.split(':').apply(lambda x: float(x[1]))
>>> df['numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
>>> df['denominator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
>>> df['denominator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[1]))
>>> df['normalized_freq'] = df['numerator'] / df['denominator']
>>> df
    surname   count frequency  ... numerator  denominator  normalized_freq
0      John  238519      1:34  ...       1.0         34.0         0.029412
1     Peter  181519      1:45  ...       1.0         45.0         0.022222
2      Paul  120946      1:67  ...       1.0         67.0         0.014925
3     David  108827      1:75  ...       1.0         75.0         0.013333
4     James   98569      1:83  ...       1.0         83.0         0.012048
..      ...     ...       ...  ...       ...          ...              ...
358    Kawi    1414   1:5,766  ...       1.0       5766.0         0.000173
359    Koko    1412   1:5,775  ...       1.0       5775.0         0.000173
360   Dokta    1409   1:5,787  ...       1.0       5787.0         0.000173
361     Gau    1407   1:5,795  ...       1.0       5795.0         0.000173
362    Karu    1401   1:5,820  ...       1.0       5820.0         0.000172

[363 rows x 8 columns]
>>> df[[c for c in df.columns if c.startswith('no')]]
     normalized_count  normalized_freq
0            0.097409         0.029412
1            0.074130         0.022222
2            0.049393         0.014925
3            0.044444         0.013333
4            0.040255         0.012048
..                ...              ...
358          0.000577         0.000173
359          0.000577         0.000173
360          0.000575         0.000173
361          0.000575         0.000173
362          0.000572         0.000172

[363 rows x 2 columns]
>>> dfs = []
... for fp in filepaths:
...     nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
... 
...     df.columns = 'rank surname count frequency'.split()
...     df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
...     df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(x: lambda x: float(x[1]))
...     df['nationality'] = nationality
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df.columns = 'rank surname count frequency'.split()
...     df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
...     df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(x: lambda x: float(x[1]))
...     df['nationality'] = nationality
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> dfs = []
... for fp in filepaths:
...     nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df.columns = 'rank surname count frequency'.split()
...     df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
...     df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(lambda x: float(x[1]))
...     df['nationality'] = nationality
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> df
     rank   surname    count frequency  freq_numerator  freq_denominator        nationality
0       1   Tesfaye  1167260      1:84             1.0              84.0          Ethiopian
1       2  Mohammed  1084839      1:90             1.0              90.0          Ethiopian
2       3  Getachew   895366     1:109             1.0             109.0          Ethiopian
3       4     Abebe   825501     1:118             1.0             118.0          Ethiopian
4       5     Girma   822765     1:119             1.0             119.0          Ethiopian
..    ...       ...      ...       ...             ...               ...                ...
358   359      Kawi     1414   1:5,766             1.0            5766.0  Papua New Guinean
359   360      Koko     1412   1:5,775             1.0            5775.0  Papua New Guinean
360   361     Dokta     1409   1:5,787             1.0            5787.0  Papua New Guinean
361   362       Gau     1407   1:5,795             1.0            5795.0  Papua New Guinean
362   363      Karu     1401   1:5,820             1.0            5820.0  Papua New Guinean

[2692 rows x 7 columns]
>>> dfs = []
... for fp in filepaths:
...     nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
...     fp = str(fp)
...     df = pd.read_html(str(fp))[-1]
...     df.columns = 'rank surname count frequency'.split()
...     df['frequency'] = df['frequency'].str.replace(',','')
...     df['freq_numerator'] = df['frequency'].str.split(':').apply(lambda x: float(x[0]))
...     df['freq_denominator'] = df['frequency'].str.split(':').apply(lambda x: float(x[1]))
...     df['nationality'] = nationality
...     dfs.append(df)
... df = pd.concat(dfs)
...
>>> df
     rank   surname    count frequency  freq_numerator  freq_denominator        nationality
0       1   Tesfaye  1167260      1:84             1.0              84.0          Ethiopian
1       2  Mohammed  1084839      1:90             1.0              90.0          Ethiopian
2       3  Getachew   895366     1:109             1.0             109.0          Ethiopian
3       4     Abebe   825501     1:118             1.0             118.0          Ethiopian
4       5     Girma   822765     1:119             1.0             119.0          Ethiopian
..    ...       ...      ...       ...             ...               ...                ...
358   359      Kawi     1414    1:5766             1.0            5766.0  Papua New Guinean
359   360      Koko     1412    1:5775             1.0            5775.0  Papua New Guinean
360   361     Dokta     1409    1:5787             1.0            5787.0  Papua New Guinean
361   362       Gau     1407    1:5795             1.0            5795.0  Papua New Guinean
362   363      Karu     1401    1:5820             1.0            5820.0  Papua New Guinean

[2692 rows x 7 columns]
>>> hist -f surname_nationality_tables_diversification.hist.py
>>> hist -o -p -f surname_nationality_tables_diversification.hist.md
>>> diversity_df = df
>>> df = load_names_from_text(dedupe=True, categories=None)
>>> diversity_df
     rank   surname    count frequency  freq_numerator  freq_denominator        nationality
0       1   Tesfaye  1167260      1:84             1.0              84.0          Ethiopian
1       2  Mohammed  1084839      1:90             1.0              90.0          Ethiopian
2       3  Getachew   895366     1:109             1.0             109.0          Ethiopian
3       4     Abebe   825501     1:118             1.0             118.0          Ethiopian
4       5     Girma   822765     1:119             1.0             119.0          Ethiopian
..    ...       ...      ...       ...             ...               ...                ...
358   359      Kawi     1414    1:5766             1.0            5766.0  Papua New Guinean
359   360      Koko     1412    1:5775             1.0            5775.0  Papua New Guinean
360   361     Dokta     1409    1:5787             1.0            5787.0  Papua New Guinean
361   362       Gau     1407    1:5795             1.0            5795.0  Papua New Guinean
362   363      Karu     1401    1:5820             1.0            5820.0  Papua New Guinean

[2692 rows x 7 columns]
>>> dftot = pd.concat([df, diversity_df])
>>> dftot
         name  category  count   rank surname frequency  freq_numerator  freq_denominator        nationality
0    Ahilelai  Nepalese      2    NaN     NaN       NaN             NaN               NaN                NaN
1     Bandana  Nepalese      1    NaN     NaN       NaN             NaN               NaN                NaN
2        Beda  Nepalese      1    NaN     NaN       NaN             NaN               NaN                NaN
3      Bhusal  Nepalese      1    NaN     NaN       NaN             NaN               NaN                NaN
4       Damai  Nepalese      1    NaN     NaN       NaN             NaN               NaN                NaN
..        ...       ...    ...    ...     ...       ...             ...               ...                ...
358       NaN       NaN   1414  359.0    Kawi    1:5766             1.0            5766.0  Papua New Guinean
359       NaN       NaN   1412  360.0    Koko    1:5775             1.0            5775.0  Papua New Guinean
360       NaN       NaN   1409  361.0   Dokta    1:5787             1.0            5787.0  Papua New Guinean
361       NaN       NaN   1407  362.0     Gau    1:5795             1.0            5795.0  Papua New Guinean
362       NaN       NaN   1401  363.0    Karu    1:5820             1.0            5820.0  Papua New Guinean

[21933 rows x 9 columns]
>>> diversity_df
     rank   surname    count frequency  freq_numerator  freq_denominator        nationality
0       1   Tesfaye  1167260      1:84             1.0              84.0          Ethiopian
1       2  Mohammed  1084839      1:90             1.0              90.0          Ethiopian
2       3  Getachew   895366     1:109             1.0             109.0          Ethiopian
3       4     Abebe   825501     1:118             1.0             118.0          Ethiopian
4       5     Girma   822765     1:119             1.0             119.0          Ethiopian
..    ...       ...      ...       ...             ...               ...                ...
358   359      Kawi     1414    1:5766             1.0            5766.0  Papua New Guinean
359   360      Koko     1412    1:5775             1.0            5775.0  Papua New Guinean
360   361     Dokta     1409    1:5787             1.0            5787.0  Papua New Guinean
361   362       Gau     1407    1:5795             1.0            5795.0  Papua New Guinean
362   363      Karu     1401    1:5820             1.0            5820.0  Papua New Guinean

[2692 rows x 7 columns]
>>> df.columns
Index(['name', 'category', 'count'], dtype='object')
>>> df.columns = 'surname nationality count'.split()
>>> dftot = pd.concat([df, diversity_df])
>>> dftot
      surname        nationality  count   rank frequency  freq_numerator  freq_denominator
0    Ahilelai           Nepalese      2    NaN       NaN             NaN               NaN
1     Bandana           Nepalese      1    NaN       NaN             NaN               NaN
2        Beda           Nepalese      1    NaN       NaN             NaN               NaN
3      Bhusal           Nepalese      1    NaN       NaN             NaN               NaN
4       Damai           Nepalese      1    NaN       NaN             NaN               NaN
..        ...                ...    ...    ...       ...             ...               ...
358      Kawi  Papua New Guinean   1414  359.0    1:5766             1.0            5766.0
359      Koko  Papua New Guinean   1412  360.0    1:5775             1.0            5775.0
360     Dokta  Papua New Guinean   1409  361.0    1:5787             1.0            5787.0
361       Gau  Papua New Guinean   1407  362.0    1:5795             1.0            5795.0
362      Karu  Papua New Guinean   1401  363.0    1:5820             1.0            5820.0

[21933 rows x 7 columns]
>>> df.to_csv('../../data/names/surnname_nationality_counts.csv.gz', compression='gzip', index=False)
>>> hist -o -p -f surname_nationality_tables_diversification.hist.md
```
