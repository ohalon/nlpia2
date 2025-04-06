>>> dfs = pd.read_html('http://www.sideroad.com/Business_Communication/politically-correct-language.html')
>>> import pandas as pd
>>> dfs = pd.read_html('http://www.sideroad.com/Business_Communication/politically-correct-language.html')
>>> dfs[0]
                          Insensitive Words & Phrases                              Possible Alternatives
0                                         Black sheep                                            Outcast
1            "Guys" (when referring to a mixed group)                              Friends; folks; group
2                 Oriental (when referring to people)  Asian (using the specific nationality, i.e. Ko...
3                            Acting like wild Indians                                     Out of control
4                 Girls (when referring to coworkers)                                              Women
5                                   Policemen/postman                        Police officer/mail carrier
6                                             Manhole                                       Utility hole
7                                            Chairman                                              Chair
8                                         Handicapped  People with special needs; people who are phys...
9                                            Retarded                         Developmentally challenged
10                                    Gifted children                                  Advanced learners
11                                               Race  Ethnicity or nationality (There is only one ra...
12              Uneducated (when referring to adults)                         Lacking a formal education
13  No culture (when referring to parts of the U.S...                           Lacking European culture
14                         The little woman; the wife                                Your wife; his wife
15                           "Don't go postal on me!"  No alternative; someone in your audience may h...
16                                      Acting blonde                                     No alternative
17                                         Old people              Seniors; "Chronologically Advantaged"
18                                 Bitchy or "PMSing"                                          Assertive
19                                        "White" lie       Lie (Calling it white does not make it okay)
20                                         Flip chart  Easel (Flip is a derogatory word referring to ...
21                                  wheel-chair bound                    A person who uses a wheel-chair
22                                           Jew down                                          Negotiate
23                                         Half-breed                                       Multi-ethnic
24                                        Blacklisted                                             Banned
25                              "Manning" the project                               Staffing the project
>>> dfs[1]
>>> from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
>>> vectorizer = CountVectorizer()
>>> vectorizer = CountVectorizer(stop_words=False)
>>> vectorizer = CountVectorizer(ngram_range(1,3), stop_words=False)
>>> vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=False)
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(1, 3), stop_words=False)
>>> df = dfs[0]
>>> df.colums = 'insensitive sensitive'.split(); vectorizer.fit(df[0]['insensitive'])
>>> df
                          Insensitive Words & Phrases                              Possible Alternatives
0                                         Black sheep                                            Outcast
1            "Guys" (when referring to a mixed group)                              Friends; folks; group
2                 Oriental (when referring to people)  Asian (using the specific nationality, i.e. Ko...
3                            Acting like wild Indians                                     Out of control
4                 Girls (when referring to coworkers)                                              Women
5                                   Policemen/postman                        Police officer/mail carrier
6                                             Manhole                                       Utility hole
7                                            Chairman                                              Chair
8                                         Handicapped  People with special needs; people who are phys...
9                                            Retarded                         Developmentally challenged
10                                    Gifted children                                  Advanced learners
11                                               Race  Ethnicity or nationality (There is only one ra...
12              Uneducated (when referring to adults)                         Lacking a formal education
13  No culture (when referring to parts of the U.S...                           Lacking European culture
14                         The little woman; the wife                                Your wife; his wife
15                           "Don't go postal on me!"  No alternative; someone in your audience may h...
16                                      Acting blonde                                     No alternative
17                                         Old people              Seniors; "Chronologically Advantaged"
18                                 Bitchy or "PMSing"                                          Assertive
19                                        "White" lie       Lie (Calling it white does not make it okay)
20                                         Flip chart  Easel (Flip is a derogatory word referring to ...
21                                  wheel-chair bound                    A person who uses a wheel-chair
22                                           Jew down                                          Negotiate
23                                         Half-breed                                       Multi-ethnic
24                                        Blacklisted                                             Banned
25                              "Manning" the project                               Staffing the project
>>> df.colums = 'insensitive sensitive'.split()
>>> df
                          Insensitive Words & Phrases                              Possible Alternatives
0                                         Black sheep                                            Outcast
1            "Guys" (when referring to a mixed group)                              Friends; folks; group
2                 Oriental (when referring to people)  Asian (using the specific nationality, i.e. Ko...
3                            Acting like wild Indians                                     Out of control
4                 Girls (when referring to coworkers)                                              Women
5                                   Policemen/postman                        Police officer/mail carrier
6                                             Manhole                                       Utility hole
7                                            Chairman                                              Chair
8                                         Handicapped  People with special needs; people who are phys...
9                                            Retarded                         Developmentally challenged
10                                    Gifted children                                  Advanced learners
11                                               Race  Ethnicity or nationality (There is only one ra...
12              Uneducated (when referring to adults)                         Lacking a formal education
13  No culture (when referring to parts of the U.S...                           Lacking European culture
14                         The little woman; the wife                                Your wife; his wife
15                           "Don't go postal on me!"  No alternative; someone in your audience may h...
16                                      Acting blonde                                     No alternative
17                                         Old people              Seniors; "Chronologically Advantaged"
18                                 Bitchy or "PMSing"                                          Assertive
19                                        "White" lie       Lie (Calling it white does not make it okay)
20                                         Flip chart  Easel (Flip is a derogatory word referring to ...
21                                  wheel-chair bound                    A person who uses a wheel-chair
22                                           Jew down                                          Negotiate
23                                         Half-breed                                       Multi-ethnic
24                                        Blacklisted                                             Banned
25                              "Manning" the project                               Staffing the project
>>> df.columns = 'insensitive sensitive'.split()
>>> df
                                          insensitive                                          sensitive
0                                         Black sheep                                            Outcast
1            "Guys" (when referring to a mixed group)                              Friends; folks; group
2                 Oriental (when referring to people)  Asian (using the specific nationality, i.e. Ko...
3                            Acting like wild Indians                                     Out of control
4                 Girls (when referring to coworkers)                                              Women
5                                   Policemen/postman                        Police officer/mail carrier
6                                             Manhole                                       Utility hole
7                                            Chairman                                              Chair
8                                         Handicapped  People with special needs; people who are phys...
9                                            Retarded                         Developmentally challenged
10                                    Gifted children                                  Advanced learners
11                                               Race  Ethnicity or nationality (There is only one ra...
12              Uneducated (when referring to adults)                         Lacking a formal education
13  No culture (when referring to parts of the U.S...                           Lacking European culture
14                         The little woman; the wife                                Your wife; his wife
15                           "Don't go postal on me!"  No alternative; someone in your audience may h...
16                                      Acting blonde                                     No alternative
17                                         Old people              Seniors; "Chronologically Advantaged"
18                                 Bitchy or "PMSing"                                          Assertive
19                                        "White" lie       Lie (Calling it white does not make it okay)
20                                         Flip chart  Easel (Flip is a derogatory word referring to ...
21                                  wheel-chair bound                    A person who uses a wheel-chair
22                                           Jew down                                          Negotiate
23                                         Half-breed                                       Multi-ethnic
24                                        Blacklisted                                             Banned
25                              "Manning" the project                               Staffing the project
>>> vectorizer.fit(df[0]['insensitive'])
>>> vectorizer.fit(df['insensitive'])
>>> df['insinsitive']
>>> df['insensitive']
0                                           Black sheep
1              "Guys" (when referring to a mixed group)
2                   Oriental (when referring to people)
3                              Acting like wild Indians
4                   Girls (when referring to coworkers)
5                                     Policemen/postman
6                                               Manhole
7                                              Chairman
8                                           Handicapped
9                                              Retarded
10                                      Gifted children
11                                                 Race
12                Uneducated (when referring to adults)
13    No culture (when referring to parts of the U.S...
14                           The little woman; the wife
15                             "Don't go postal on me!"
16                                        Acting blonde
17                                           Old people
18                                   Bitchy or "PMSing"
19                                          "White" lie
20                                           Flip chart
21                                    wheel-chair bound
22                                             Jew down
23                                           Half-breed
24                                          Blacklisted
25                                "Manning" the project
Name: insensitive, dtype: object
>>> vectorizer.fit(df['insensitive'])
>>> vectorizer.fit([str(x) for x in df['insensitive']])
>>> df.info()
>>> len(df)
26
>>> df['insensitive'].str.len()
0      11
1      40
2      35
3      24
4      35
5      17
6       7
7       8
8      11
9       8
10     15
11      4
12     37
13    106
14     26
15     24
16     13
17     10
18     18
19     11
20     10
21     17
22      8
23     10
24     11
25     21
Name: insensitive, dtype: int64
>>> df['insensitive'].str.str()
>>> df['insensitive'].str.__str__()
'<pandas.core.strings.accessor.StringMethods object at 0x7f1468606b90>'
>>> [str(s) for s in df['insensitive']]
['Black sheep',
 '"Guys" (when referring to a mixed group)',
 'Oriental (when referring to people)',
 'Acting like wild Indians',
 'Girls (when referring to coworkers)',
 'Policemen/postman',
 'Manhole',
 'Chairman',
 'Handicapped',
 'Retarded',
 'Gifted children',
 'Race',
 'Uneducated (when referring to adults)',
 'No culture (when referring to parts of the U.S. where the opera and the theater are scarce or nonexistent)',
 'The little woman; the wife',
 '"Don\'t go postal on me!"',
 'Acting blonde',
 'Old people',
 'Bitchy or "PMSing"',
 '"White" lie',
 'Flip chart',
 'wheel-chair bound',
 'Jew down',
 'Half-breed',
 'Blacklisted',
 '"Manning" the project']
>>> texts = [str(s) for s in df['insensitive']]
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(1, 3), stop_words=None)
>>> vectorizer.fit(texts)
CountVectorizer(max_df=0.5, ngram_range=(1, 3))
>>> vectorizer.transform(texts)
<26x154 sparse matrix of type '<class 'numpy.int64'>'
	with 183 stored elements in Compressed Sparse Row format>
>>> countvecs = vectorizer.transform(texts).todense()
>>> countvecs
matrix([[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]])
>>> pd.DataFrame(countvecs, columns=vectorizer.get_feature_names())
    acting  acting blonde  acting like  acting like wild  adults  ...  wild  wild indians  woman  woman the  woman the wife
0        0              0            0                 0       0  ...     0             0      0          0               0
1        0              0            0                 0       0  ...     0             0      0          0               0
2        0              0            0                 0       0  ...     0             0      0          0               0
3        1              0            1                 1       0  ...     1             1      0          0               0
4        0              0            0                 0       0  ...     0             0      0          0               0
5        0              0            0                 0       0  ...     0             0      0          0               0
6        0              0            0                 0       0  ...     0             0      0          0               0
7        0              0            0                 0       0  ...     0             0      0          0               0
8        0              0            0                 0       0  ...     0             0      0          0               0
9        0              0            0                 0       0  ...     0             0      0          0               0
10       0              0            0                 0       0  ...     0             0      0          0               0
11       0              0            0                 0       0  ...     0             0      0          0               0
12       0              0            0                 0       1  ...     0             0      0          0               0
13       0              0            0                 0       0  ...     0             0      0          0               0
14       0              0            0                 0       0  ...     0             0      1          1               1
15       0              0            0                 0       0  ...     0             0      0          0               0
16       1              1            0                 0       0  ...     0             0      0          0               0
17       0              0            0                 0       0  ...     0             0      0          0               0
18       0              0            0                 0       0  ...     0             0      0          0               0
19       0              0            0                 0       0  ...     0             0      0          0               0
20       0              0            0                 0       0  ...     0             0      0          0               0
21       0              0            0                 0       0  ...     0             0      0          0               0
22       0              0            0                 0       0  ...     0             0      0          0               0
23       0              0            0                 0       0  ...     0             0      0          0               0
24       0              0            0                 0       0  ...     0             0      0          0               0
25       0              0            0                 0       0  ...     0             0      0          0               0

[26 rows x 154 columns]
>>> pd.DataFrame(countvecs, columns=vectorizer.get_feature_names()).T
                  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
acting             0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
acting blonde      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
acting like        0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
acting like wild   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
adults             0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
...               ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
wild               0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
wild indians       0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
woman              0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
woman the          0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
woman the wife     0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0

[154 rows x 26 columns]
>>> cv = pd.DataFrame(countvecs, columns=vectorizer.get_feature_names())
>>> cv.T
                  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
acting             0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
acting blonde      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
acting like        0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
acting like wild   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
adults             0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
...               ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
wild               0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
wild indians       0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
woman              0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
woman the          0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
woman the wife     0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0

[154 rows x 26 columns]
>>> for t in texts:
...     if 'wild' in t:
...         print(t)
...
>>> query = 'wild acting'
>>> vectorizer.transform(query)
>>> vectorizer.transform([query])
<1x154 sparse matrix of type '<class 'numpy.int64'>'
	with 2 stored elements in Compressed Sparse Row format>
>>> vectorizer.transform([query]).todense()
matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0]])
>>> vectorizer.transform([query]).todense()[0]
matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0]])
>>> m = vectorizer.transform([query]).todense()
>>> m.flatten()
matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0]])
>>> m.tolist()
[[1,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0]]
>>> m.tolist()[0]
[1,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 1,
 0,
 0,
 0,
 0]
>>> pd.np.array(m)[0]
array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
>>> m = vectorizer.transform([query]).toarray()
>>> m
array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
>>> m[0]
array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
>>> qv = m[0]
>>> cv
    acting  acting blonde  acting like  acting like wild  adults  and  and the  and the theater  ...  white  white lie  wife  wild  wild indians  woman  woman the  woman the wife
0        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
1        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
2        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
3        1              0            1                 1       0    0        0                0  ...      0          0     0     1             1      0          0               0
4        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
5        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
6        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
7        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
8        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
9        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
10       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
11       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
12       0              0            0                 0       1    0        0                0  ...      0          0     0     0             0      0          0               0
13       0              0            0                 0       0    1        1                1  ...      0          0     0     0             0      0          0               0
14       0              0            0                 0       0    0        0                0  ...      0          0     1     0             0      1          1               1
15       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
16       1              1            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
17       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
18       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
19       0              0            0                 0       0    0        0                0  ...      1          1     0     0             0      0          0               0
20       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
21       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
22       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
23       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
24       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
25       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0

[26 rows x 154 columns]
>>> cv.dot(qv)
0     0
1     0
2     0
3     2
4     0
5     0
6     0
7     0
8     0
9     0
10    0
11    0
12    0
13    0
14    0
15    0
16    1
17    0
18    0
19    0
20    0
21    0
22    0
23    0
24    0
25    0
dtype: int64
>>> cv.dot(qv).sort()
>>> cv.dot(qv).sortvalues()
>>> cv.dot(qv).sort_values()
0     0
23    0
22    0
21    0
20    0
19    0
18    0
17    0
15    0
14    0
13    0
24    0
12    0
10    0
9     0
8     0
7     0
6     0
5     0
4     0
2     0
1     0
11    0
25    0
16    1
3     2
dtype: int64
>>> cv.dot(qv).sort_values(ascending=False)
3     2
16    1
0     0
14    0
24    0
23    0
22    0
21    0
20    0
19    0
18    0
17    0
15    0
13    0
1     0
12    0
11    0
10    0
9     0
8     0
7     0
6     0
5     0
4     0
2     0
25    0
dtype: int64
>>> cv.dot(qv).sort_values(ascending=False)[0]
0
>>> cv.dot(qv).sort_values(ascending=False).index[0]
3
>>> texts[cv.dot(qv).sort_values(ascending=False).index[0]]
'Acting like wild Indians'
>>> history
>>> m
array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
>>> cv
    acting  acting blonde  acting like  acting like wild  adults  and  and the  and the theater  ...  white  white lie  wife  wild  wild indians  woman  woman the  woman the wife
0        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
1        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
2        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
3        1              0            1                 1       0    0        0                0  ...      0          0     0     1             1      0          0               0
4        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
5        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
6        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
7        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
8        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
9        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
10       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
11       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
12       0              0            0                 0       1    0        0                0  ...      0          0     0     0             0      0          0               0
13       0              0            0                 0       0    1        1                1  ...      0          0     0     0             0      0          0               0
14       0              0            0                 0       0    0        0                0  ...      0          0     1     0             0      1          1               1
15       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
16       1              1            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
17       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
18       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
19       0              0            0                 0       0    0        0                0  ...      1          1     0     0             0      0          0               0
20       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
21       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
22       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
23       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
24       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
25       0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0

[26 rows x 154 columns]
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,), stop_words=None, analyzer=list)
>>> vectorizer.fit(texts)
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,), stop_words=None, analyzer='char')
>>> vectorizer.fit(texts)
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=[3], stop_words=None, analyzer='char')
>>> vectorizer.fit(texts)
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,3), stop_words=None, analyzer='char')
>>> vectorizer.fit(texts)
CountVectorizer(analyzer='char', max_df=0.5, ngram_range=(3, 3))
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,3), stop_words=None, analyzer=list)
>>> vectorizer.fit(texts)
CountVectorizer(analyzer=<class 'list'>, max_df=0.5, ngram_range=(3, 3))
>>> vectorizer.transform(texts)
<26x41 sparse matrix of type '<class 'numpy.int64'>'
	with 146 stored elements in Compressed Sparse Row format>
>>> vectorizer.transform(texts).toarray()
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 2, 0, ..., 1, 1, 1],
       [0, 0, 0, ..., 1, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 2, 0, ..., 0, 0, 0]])
>>> trgcounts = pd.DataFrame(vectorizer.transform(texts).toarray(), columns=vectorizer.get_feature_names())
>>> trgcounts
    !  "  '  (  )  -  .  /  ;  A  B  C  D  F  G  H  I  J  M  N  O  P  R  S  T  U  W  b  d  f  g  j  k  m  p   r  s  u  w  x  y
0   0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1   0  1  0  0  0  0
1   0  2  0  1  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  2  0  0  1  1   4  1  2  1  1  1
2   0  0  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  1  1  0  0  0  2   4  0  0  1  0  0
3   0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  2  0  1  0  1  0  0   0  1  0  1  0  0
4   0  0  0  1  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0  1  0  0   6  2  0  2  0  0
5   0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  2  1   0  1  0  0  0  0
6   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   0  0  0  0  0  0
7   0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0   1  0  0  0  0  0
8   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  2   0  0  0  0  0  0
9   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  2  0  0  0  0  0  0   1  0  0  0  0  0
10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  2  1  0  0  0  0  0   1  0  0  0  0  0
11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0   0  0  0  0  0  0
12  0  0  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  3  1  1  0  0  0  0   3  1  2  1  0  0
13  0  0  0  1  1  0  2  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  1  0  1  0  0  1  2  1  0  0  0  2  11  3  2  2  1  0
14  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  1  0  0  0  1  0   0  0  0  2  0  0
15  1  2  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  1  1   0  1  0  0  0  0
16  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0  1  0  0  0  0   0  0  0  0  0  0
17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  0  0  0  2   0  0  0  0  0  0
18  0  2  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0  0  1  0  1  0  0  0  0  0  0  1  0  0  0  0   1  0  0  0  0  1
19  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0   0  0  0  0  0  0
20  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1   1  0  0  0  0  0
21  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  0  0  0  0  0  0   1  0  1  1  0  0
22  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0   0  0  0  2  0  0
23  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  1  1  0  0  0  0  0   1  0  0  0  0  0
24  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  1  0  0   0  1  0  0  0  0
25  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  1  0  0  1   1  0  0  0  0  0
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,3), stop_words=None, analyzer='char')
>>> CountVectorizer?
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,3), stop_words=None, tokenizer=list)
>>> vectorizer.fit(texts)
CountVectorizer(max_df=0.5, ngram_range=(3, 3), tokenizer=<class 'list'>)
>>> trgcounts = pd.DataFrame(vectorizer.transform(texts).toarray(), columns=vectorizer.get_feature_names())
>>> trgcounts
      " p    ( w    a      a d    a n    a r    b l    b o    c h    c o    c u    d o  ...  u y s  w   d  w h e  w h i  w i f  w i l  w o m  w o r  x e d  x i s  y   o  y s "
0       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
1       0      1      1      0      0      0      0      0      0      0      0      0  ...      1      0      1      0      0      0      0      0      1      0      0      1
2       0      1      0      0      0      0      0      0      0      0      0      0  ...      0      0      1      0      0      0      0      0      0      0      0      0
3       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      1      0      0      0      0      0      0
4       0      1      0      0      0      0      0      0      0      1      0      0  ...      0      0      1      0      0      0      0      1      0      0      0      0
5       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
6       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
7       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
8       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
9       0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
10      0      0      0      0      0      0      0      0      1      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
11      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
12      0      1      0      1      0      0      0      0      0      0      0      0  ...      0      0      1      0      0      0      0      0      0      0      0      0
13      0      1      0      0      1      1      0      0      0      0      1      0  ...      0      0      2      0      0      0      0      0      0      1      0      0
14      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      1      0      1      0      0      0      0      0
15      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
16      0      0      0      0      0      0      1      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
17      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
18      1      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      1      0
19      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      1      0      0      0      0      0      0      0      0
20      0      0      0      0      0      0      0      0      1      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
21      0      0      0      0      0      0      0      1      0      0      0      0  ...      0      0      1      0      0      0      0      0      0      0      0      0
22      0      0      0      0      0      0      0      0      0      0      0      1  ...      0      1      0      0      0      0      0      0      0      0      0      0
23      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
24      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0
25      0      0      0      0      0      0      0      0      0      0      0      0  ...      0      0      0      0      0      0      0      0      0      0      0      0

[26 rows x 334 columns]
>>> trgcounts.columns[0]
'  " p'
>>> vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3,3), stop_words=None, analyzer='char')
>>> vectorizer.fit(texts)
CountVectorizer(analyzer='char', max_df=0.5, ngram_range=(3, 3))
>>> trgcounts = pd.DataFrame(vectorizer.transform(texts).toarray(), columns=vectorizer.get_feature_names())
>>> trgcounts.columns[0]
' "p'
>>> vectorizer.transform([qyery])
>>> vectorizer.transform([query])
<1x334 sparse matrix of type '<class 'numpy.int64'>'
	with 7 stored elements in Compressed Sparse Row format>
>>> qv = vectorizer.transform([query]).toarray()[0]
>>> trgcounts.dot(qv)
0     0
1     1
2     1
3     7
4     1
5     0
6     0
7     0
8     0
9     0
10    1
11    0
12    1
13    1
14    0
15    0
16    4
17    1
18    1
19    0
20    0
21    0
22    0
23    0
24    0
25    1
dtype: int64
>>> query
'wild acting'
>>> query = 'wld actng'
>>> trgcounts.dot(qv)
0     0
1     1
2     1
3     7
4     1
5     0
6     0
7     0
8     0
9     0
10    1
11    0
12    1
13    1
14    0
15    0
16    4
17    1
18    1
19    0
20    0
21    0
22    0
23    0
24    0
25    1
dtype: int64
>>> qv = vectorizer.transform([query]).toarray()[0]
>>> trgcounts.dot(qv)
0     0
1     0
2     0
3     2
4     0
5     0
6     0
7     0
8     0
9     0
10    0
11    0
12    0
13    0
14    0
15    0
16    1
17    1
18    0
19    0
20    0
21    0
22    0
23    0
24    0
25    0
dtype: int64
>>> query = 'wild actng'
>>> query = 'wild actin'
>>> qv = vectorizer.transform([query]).toarray()[0]
>>> trgcounts.dot(qv)
0     0
1     0
2     0
3     6
4     0
5     0
6     0
7     0
8     0
9     0
10    1
11    0
12    0
13    0
14    0
15    0
16    3
17    1
18    0
19    0
20    0
21    0
22    0
23    0
24    0
25    0
dtype: int64
>>> trgcounts.sum()
 "p    1
 (w    5
 a     1
 ad    1
 an    1
      ..
wor    1
xed    1
xis    1
y o    1
ys"    1
Length: 334, dtype: int64
>>> counts = trgcounts.sum()
>>> counts.index
Index([' "p', ' (w', ' a ', ' ad', ' an', ' ar', ' bl', ' bo', ' ch', ' co',
       ...
       'whe', 'whi', 'wif', 'wil', 'wom', 'wor', 'xed', 'xis', 'y o', 'ys"'],
      dtype='object', length=334)
>>> from collections import Counter
... c = Counter()
... for trg, num in zip(counts.index, counts.values):
...     trg[:2]
...
>>> counts.index.str.len()
Int64Index([3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            ...
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
           dtype='int64', length=334)
>>> counts.index.str[:2]
Index([' "', ' (', ' a', ' a', ' a', ' a', ' b', ' b', ' c', ' c',
       ...
       'wh', 'wh', 'wi', 'wi', 'wo', 'wo', 'xe', 'xi', 'y ', 'ys'],
      dtype='object', length=334)
>>> from collections import Counter
... c = Counter()
... for bg, num in zip(counts.index.str[:2], counts.values):
...     c += Counter(zip(*[bg, num]))
...
>>> from collections import Counter
... c = Counter()
... for bg, num in zip(counts.index.str[:2], counts.values):
...     c += Counter(bg, num)
...
>>> from collections import Counter
... c = Counter()
... for bg, num in zip(counts.index.str[:2], counts.values):
...     c += Counter(zip(bg, num))
...
>>> num
1
>>> bg
' "'
>>> from collections import Counter
... c = Counter()
... for bg, num in zip(counts.index.str[:2], counts.values):
...     c += Counter((bg, num))
...
>>> c
Counter({' "': 1,
         1: 268,
         ' (': 1,
         5: 16,
         ' a': 4,
         ' b': 2,
         ' c': 3,
         2: 37,
         ' d': 1,
         ' g': 2,
         ' i': 1,
         ' l': 1,
         3: 7,
         ' m': 2,
         ' n': 1,
         ' o': 4,
         ' p': 4,
         ' r': 1,
         ' s': 2,
         ' t': 2,
         6: 2,
         ' u': 1,
         ' w': 3,
         '" ': 3,
         '"d': 1,
         '"g': 1,
         '"m': 1,
         '"p': 1,
         '"w': 1,
         "'t": 1,
         '(w': 1,
         '-b': 1,
         '-c': 1,
         '. ': 1,
         '.s': 1,
         '/p': 1,
         '; ': 1,
         'a ': 2,
         'ac': 3,
         'ad': 1,
         'ai': 1,
         'al': 2,
         'an': 5,
         'ap': 1,
         'ar': 4,
         'at': 1,
         'bi': 1,
         'bl': 2,
         'bo': 1,
         'br': 1,
         'ca': 3,
         'ce': 2,
         'ch': 3,
         'ck': 2,
         'co': 1,
         'ct': 1,
         'cu': 1,
         'd ': 6,
         'de': 1,
         'di': 2,
         'do': 2,
         'dr': 1,
         'du': 2,
         'e ': 8,
         'e!': 1,
         'e"': 1,
         'ea': 1,
         'ec': 1,
         'ed': 2,
         'ee': 3,
         'ef': 1,
         'el': 1,
         'em': 1,
         'en': 3,
         'eo': 1,
         'er': 5,
         'et': 1,
         'ew': 1,
         'ex': 1,
         'f ': 1,
         'f-': 1,
         'fe': 1,
         'fl': 1,
         'ft': 1,
         'g ': 3,
         'g"': 1,
         'gi': 2,
         'go': 1,
         'gr': 1,
         'gu': 1,
         'ha': 4,
         'he': 5,
         'hi': 2,
         'ho': 1,
         'hy': 1,
         'ia': 1,
         'ic': 2,
         'ie': 1,
         'if': 2,
         'ik': 1,
         'il': 1,
         'in': 2,
         9: 1,
         'ip': 1,
         'ir': 3,
         'is': 1,
         'it': 3,
         'ix': 1,
         'je': 2,
         'k ': 1,
         'ke': 2,
         'kl': 1,
         'l ': 2,
         'l-': 1,
         'la': 1,
         'ld': 2,
         'le': 2,
         'lf': 1,
         'li': 6,
         'lo': 1,
         'ls': 1,
         'lt': 2,
         'ma': 1,
         'me': 2,
         'mi': 1,
         'ms': 1,
         'n ': 2,
         "n'": 1,
         'n/': 1,
         'n;': 1,
         'nd': 3,
         'ne': 2,
         'ng': 2,
         7: 3,
         'nh': 1,
         'ni': 1,
         'nn': 1,
         'no': 2,
         'nt': 2,
         'o ': 3,
         'of': 1,
         'oj': 1,
         'ol': 3,
         'om': 1,
         'on': 4,
         'op': 2,
         'or': 3,
         'os': 1,
         'ou': 2,
         'ow': 2,
         'p ': 1,
         'pa': 1,
         'pe': 3,
         'pl': 1,
         'pm': 1,
         'po': 2,
         'pp': 1,
         'pr': 1,
         'r ': 4,
         'ra': 2,
         'rc': 1,
         'rd': 1,
         're': 5,
         'ri': 2,
         'rk': 1,
         'rl': 1,
         'rm': 1,
         'ro': 2,
         'rr': 1,
         'rs': 1,
         'rt': 1,
         's ': 2,
         's"': 1,
         's.': 1,
         'sc': 1,
         'sh': 1,
         'si': 1,
         'st': 3,
         't ': 1,
         'ta': 2,
         'tc': 1,
         'te': 4,
         'th': 1,
         'ti': 1,
         'tl': 1,
         'tm': 1,
         'to': 1,
         'ts': 2,
         'tt': 1,
         'tu': 1,
         'u.': 1,
         'uc': 1,
         'ul': 1,
         'un': 2,
         'up': 1,
         'ur': 1,
         'uy': 1,
         'w ': 1,
         'wh': 2,
         'wi': 2,
         'wo': 2,
         'xe': 1,
         'xi': 1,
         'y ': 1,
         'ys': 1})
>>> counds.sum()
>>> counts.sum()
485
>>> prob = pd.Series(c) / counts.sum()
>>> seed = 'he'
>>> while
>>> texts
['Black sheep',
 '"Guys" (when referring to a mixed group)',
 'Oriental (when referring to people)',
 'Acting like wild Indians',
 'Girls (when referring to coworkers)',
 'Policemen/postman',
 'Manhole',
 'Chairman',
 'Handicapped',
 'Retarded',
 'Gifted children',
 'Race',
 'Uneducated (when referring to adults)',
 'No culture (when referring to parts of the U.S. where the opera and the theater are scarce or nonexistent)',
 'The little woman; the wife',
 '"Don\'t go postal on me!"',
 'Acting blonde',
 'Old people',
 'Bitchy or "PMSing"',
 '"White" lie',
 'Flip chart',
 'wheel-chair bound',
 'Jew down',
 'Half-breed',
 'Blacklisted',
 '"Manning" the project']
>>> for i in range(30):
...     seed +=
...
>>> for i in range(30):
...     try:
...         seed += prob.get(seed[-2:])
...     except ValueError:
...         break
...
>>> for i in range(30):
...     try:
...         seed += prob.get(seed[-2:])
...     except TypeError:
...         break
...
>>> seed
'he'
>>> prob.get('he')
0.010309278350515464
>>> prob
 "    0.002062
1     0.552577
 (    0.002062
5     0.032990
 a    0.008247
        ...
wo    0.004124
xe    0.002062
xi    0.002062
y     0.002062
ys    0.002062
Length: 205, dtype: float64
>>> from collections import Counter
... c = Counter()
... for bg, num, thirdchar in zip(counts.index.str[:2], counts.values, counts.index.str[-1]):
...     c += Counter((bg, thirdchar), num)
...
>>> from collections import Counter
... c = Counter()
... for bg, num, thirdchar in zip(counts.index.str[:2], counts.values, counts.index.str[-1]):
...     c += Counter(((bg, thirdchar), num))
...
>>> c
Counter({(' "', 'p'): 1,
         1: 268,
         (' (', 'w'): 1,
         5: 16,
         (' a', ' '): 1,
         (' a', 'd'): 1,
         (' a', 'n'): 1,
         (' a', 'r'): 1,
         (' b', 'l'): 1,
         (' b', 'o'): 1,
         (' c', 'h'): 1,
         2: 37,
         (' c', 'o'): 1,
         (' c', 'u'): 1,
         (' d', 'o'): 1,
         (' g', 'o'): 1,
         (' g', 'r'): 1,
         (' i', 'n'): 1,
         (' l', 'i'): 1,
         3: 7,
         (' m', 'e'): 1,
         (' m', 'i'): 1,
         (' n', 'o'): 1,
         (' o', 'f'): 1,
         (' o', 'n'): 1,
         (' o', 'p'): 1,
         (' o', 'r'): 1,
         (' p', 'a'): 1,
         (' p', 'e'): 1,
         (' p', 'o'): 1,
         (' p', 'r'): 1,
         (' r', 'e'): 1,
         (' s', 'c'): 1,
         (' s', 'h'): 1,
         (' t', 'h'): 1,
         6: 2,
         (' t', 'o'): 1,
         (' u', '.'): 1,
         (' w', 'h'): 1,
         (' w', 'i'): 1,
         (' w', 'o'): 1,
         ('" ', '('): 1,
         ('" ', 'l'): 1,
         ('" ', 't'): 1,
         ('"d', 'o'): 1,
         ('"g', 'u'): 1,
         ('"m', 'a'): 1,
         ('"p', 'm'): 1,
         ('"w', 'h'): 1,
         ("'t", ' '): 1,
         ('(w', 'h'): 1,
         ('-b', 'r'): 1,
         ('-c', 'h'): 1,
         ('. ', 'w'): 1,
         ('.s', '.'): 1,
         ('/p', 'o'): 1,
         ('; ', 't'): 1,
         ('a ', 'a'): 1,
         ('a ', 'm'): 1,
         ('ac', 'e'): 1,
         ('ac', 'k'): 1,
         ('ac', 't'): 1,
         ('ad', 'u'): 1,
         ('ai', 'r'): 1,
         ('al', ' '): 1,
         ('al', 'f'): 1,
         ('an', ';'): 1,
         ('an', 'd'): 1,
         ('an', 'h'): 1,
         ('an', 'n'): 1,
         ('an', 's'): 1,
         ('ap', 'p'): 1,
         ('ar', 'c'): 1,
         ('ar', 'd'): 1,
         ('ar', 'e'): 1,
         ('ar', 't'): 1,
         ('at', 'e'): 1,
         ('bi', 't'): 1,
         ('bl', 'a'): 1,
         ('bl', 'o'): 1,
         ('bo', 'u'): 1,
         ('br', 'e'): 1,
         ('ca', 'p'): 1,
         ('ca', 'r'): 1,
         ('ca', 't'): 1,
         ('ce', ' '): 1,
         ('ce', 'm'): 1,
         ('ch', 'a'): 1,
         ('ch', 'i'): 1,
         ('ch', 'y'): 1,
         ('ck', ' '): 1,
         ('ck', 'l'): 1,
         ('co', 'w'): 1,
         ('ct', 'i'): 1,
         ('cu', 'l'): 1,
         ('d ', '('): 1,
         ('d ', 'c'): 1,
         ('d ', 'g'): 1,
         ('d ', 'i'): 1,
         ('d ', 'p'): 1,
         ('d ', 't'): 1,
         ('de', 'd'): 1,
         ('di', 'a'): 1,
         ('di', 'c'): 1,
         ('do', 'n'): 1,
         ('do', 'w'): 1,
         ('dr', 'e'): 1,
         ('du', 'c'): 1,
         ('du', 'l'): 1,
         ('e ', '('): 1,
         ('e ', 'l'): 1,
         ('e ', 'o'): 1,
         ('e ', 'p'): 1,
         ('e ', 's'): 1,
         ('e ', 't'): 1,
         ('e ', 'u'): 1,
         ('e ', 'w'): 1,
         ('e!', '"'): 1,
         ('e"', ' '): 1,
         ('ea', 't'): 1,
         ('ec', 't'): 1,
         ('ed', ' '): 1,
         ('ed', 'u'): 1,
         ('ee', 'd'): 1,
         ('ee', 'l'): 1,
         ('ee', 'p'): 1,
         ('ef', 'e'): 1,
         ('el', '-'): 1,
         ('em', 'e'): 1,
         ('en', ' '): 1,
         ('en', '/'): 1,
         ('en', 't'): 1,
         ('eo', 'p'): 1,
         ('er', ' '): 1,
         ('er', 'a'): 1,
         ('er', 'e'): 1,
         ('er', 'r'): 1,
         ('er', 's'): 1,
         ('et', 'a'): 1,
         ('ew', ' '): 1,
         ('ex', 'i'): 1,
         ('f ', 't'): 1,
         ('f-', 'b'): 1,
         ('fe', 'r'): 1,
         ('fl', 'i'): 1,
         ('ft', 'e'): 1,
         ('g ', 'b'): 1,
         ('g ', 'l'): 1,
         ('g ', 't'): 1,
         ('g"', ' '): 1,
         ('gi', 'f'): 1,
         ('gi', 'r'): 1,
         ('go', ' '): 1,
         ('gr', 'o'): 1,
         ('gu', 'y'): 1,
         ('ha', 'i'): 1,
         ('ha', 'l'): 1,
         ('ha', 'n'): 1,
         ('ha', 'r'): 1,
         ('he', ' '): 1,
         ('he', 'a'): 1,
         ('he', 'e'): 1,
         ('he', 'n'): 1,
         ('he', 'r'): 1,
         ('hi', 'l'): 1,
         ('hi', 't'): 1,
         ('ho', 'l'): 1,
         ('hy', ' '): 1,
         ('ia', 'n'): 1,
         ('ic', 'a'): 1,
         ('ic', 'e'): 1,
         ('ie', 'n'): 1,
         ('if', 'e'): 1,
         ('if', 't'): 1,
         ('ik', 'e'): 1,
         ('il', 'd'): 1,
         ('in', 'd'): 1,
         ('in', 'g'): 1,
         9: 1,
         ('ip', ' '): 1,
         ('ir', ' '): 1,
         ('ir', 'l'): 1,
         ('ir', 'm'): 1,
         ('is', 't'): 1,
         ('it', 'c'): 1,
         ('it', 'e'): 1,
         ('it', 't'): 1,
         ('ix', 'e'): 1,
         ('je', 'c'): 1,
         ('je', 'w'): 1,
         ('k ', 's'): 1,
         ('ke', ' '): 1,
         ('ke', 'r'): 1,
         ('kl', 'i'): 1,
         ('l ', '('): 1,
         ('l ', 'o'): 1,
         ('l-', 'c'): 1,
         ('la', 'c'): 1,
         ('ld', ' '): 1,
         ('ld', 'r'): 1,
         ('le', ' '): 1,
         ('le', ')'): 1,
         ('lf', '-'): 1,
         ('li', 'c'): 1,
         ('li', 'e'): 1,
         ('li', 'k'): 1,
         ('li', 'p'): 1,
         ('li', 's'): 1,
         ('li', 't'): 1,
         ('lo', 'n'): 1,
         ('ls', ' '): 1,
         ('lt', 's'): 1,
         ('lt', 'u'): 1,
         ('ma', 'n'): 1,
         ('me', '!'): 1,
         ('me', 'n'): 1,
         ('mi', 'x'): 1,
         ('ms', 'i'): 1,
         ('n ', 'm'): 1,
         ('n ', 'r'): 1,
         ("n'", 't'): 1,
         ('n/', 'p'): 1,
         ('n;', ' '): 1,
         ('nd', ' '): 1,
         ('nd', 'e'): 1,
         ('nd', 'i'): 1,
         ('ne', 'd'): 1,
         ('ne', 'x'): 1,
         ('ng', ' '): 1,
         7: 3,
         ('ng', '"'): 1,
         ('nh', 'o'): 1,
         ('ni', 'n'): 1,
         ('nn', 'i'): 1,
         ('no', ' '): 1,
         ('no', 'n'): 1,
         ('nt', ')'): 1,
         ('nt', 'a'): 1,
         ('o ', 'a'): 1,
         ('o ', 'c'): 1,
         ('o ', 'p'): 1,
         ('of', ' '): 1,
         ('oj', 'e'): 1,
         ('ol', 'd'): 1,
         ('ol', 'e'): 1,
         ('ol', 'i'): 1,
         ('om', 'a'): 1,
         ('on', ' '): 1,
         ('on', "'"): 1,
         ('on', 'd'): 1,
         ('on', 'e'): 1,
         ('op', 'e'): 1,
         ('op', 'l'): 1,
         ('or', ' '): 1,
         ('or', 'i'): 1,
         ('or', 'k'): 1,
         ('os', 't'): 1,
         ('ou', 'n'): 1,
         ('ou', 'p'): 1,
         ('ow', 'n'): 1,
         ('ow', 'o'): 1,
         ('p ', 'c'): 1,
         ('pa', 'r'): 1,
         ('pe', 'd'): 1,
         ('pe', 'o'): 1,
         ('pe', 'r'): 1,
         ('pl', 'e'): 1,
         ('pm', 's'): 1,
         ('po', 'l'): 1,
         ('po', 's'): 1,
         ('pp', 'e'): 1,
         ('pr', 'o'): 1,
         ('r ', '"'): 1,
         ('r ', 'a'): 1,
         ('r ', 'b'): 1,
         ('r ', 'n'): 1,
         ('ra', ' '): 1,
         ('ra', 'c'): 1,
         ('rc', 'e'): 1,
         ('rd', 'e'): 1,
         ('re', ' '): 1,
         ('re', 'e'): 1,
         ('re', 'f'): 1,
         ('re', 'n'): 1,
         ('re', 't'): 1,
         ('ri', 'e'): 1,
         ('ri', 'n'): 1,
         ('rk', 'e'): 1,
         ('rl', 's'): 1,
         ('rm', 'a'): 1,
         ('ro', 'j'): 1,
         ('ro', 'u'): 1,
         ('rr', 'i'): 1,
         ('rs', ')'): 1,
         ('rt', 's'): 1,
         ('s ', '('): 1,
         ('s ', 'o'): 1,
         ('s"', ' '): 1,
         ('s.', ' '): 1,
         ('sc', 'a'): 1,
         ('sh', 'e'): 1,
         ('si', 'n'): 1,
         ('st', 'a'): 1,
         ('st', 'e'): 1,
         ('st', 'm'): 1,
         ('t ', 'g'): 1,
         ('ta', 'l'): 1,
         ('ta', 'r'): 1,
         ('tc', 'h'): 1,
         ('te', '"'): 1,
         ('te', 'd'): 1,
         ('te', 'n'): 1,
         ('te', 'r'): 1,
         ('th', 'e'): 1,
         ('ti', 'n'): 1,
         ('tl', 'e'): 1,
         ('tm', 'a'): 1,
         ('to', ' '): 1,
         ('ts', ' '): 1,
         ('ts', ')'): 1,
         ('tt', 'l'): 1,
         ('tu', 'r'): 1,
         ('u.', 's'): 1,
         ('uc', 'a'): 1,
         ('ul', 't'): 1,
         ('un', 'd'): 1,
         ('un', 'e'): 1,
         ('up', ')'): 1,
         ('ur', 'e'): 1,
         ('uy', 's'): 1,
         ('w ', 'd'): 1,
         ('wh', 'e'): 1,
         ('wh', 'i'): 1,
         ('wi', 'f'): 1,
         ('wi', 'l'): 1,
         ('wo', 'm'): 1,
         ('wo', 'r'): 1,
         ('xe', 'd'): 1,
         ('xi', 's'): 1,
         ('y ', 'o'): 1,
         ('ys', '"'): 1})
>>> [bigram for bigram, onegram in c.keys()]
>>> [bigram for bigram, onegram in zip(*c.keys())]
>>> history -o -p -f 'char_trigram_index_and_language_model.ipy'
