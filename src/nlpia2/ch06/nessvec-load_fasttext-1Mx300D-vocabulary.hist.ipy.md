>>> import nessvec
>>> from nessvec.files import load_fasttext
>>> df = load_fasttext()
>>> df['token'] = df.index.values
>>> df['token'].str.contains(' ').sum()
0
>>> df['token'].str.contains('_').sum()
0
>>> df['token'].str.contains('-').sum()
141666
>>> df['token'].str.contains(',')]
>>> df[df['token'].str.contains('-')].head()
                0       1       2       3       4       5       6       7       8  ...     292     293     294     295     296     297     298     299      token
-         -0.0092 -0.0478 -0.0380  0.0125  0.0175 -0.0282  0.1159  0.0997 -0.0636  ... -0.0463 -0.0377 -0.0127  0.0170 -0.0551  0.0041 -0.0209 -0.0178          -
--        -0.0524 -0.0049 -0.0400  0.0253  0.0142 -0.0445  0.0630  0.0699  0.0124  ... -0.0496  0.0533 -0.1793  0.0452 -0.0273  0.1652 -0.0144 -0.0744         --
long-term  0.0863 -0.0190 -0.0093  0.0286  0.0725  0.1231 -0.0786  0.1026 -0.2255  ...  0.0948  0.0758  0.0015  0.1331 -0.0873  0.1999 -0.0986 -0.1016  long-term
so-called  0.0731 -0.0898  0.0593  0.0246  0.0074  0.1135 -0.0081  0.0289 -0.0401  ... -0.0643 -0.0948 -0.1312  0.0199 -0.0594  0.1277  0.0752 -0.0099  so-called
e-mail    -0.1670 -0.0579  0.1965  0.0349  0.1167 -0.0685  0.0254  0.2079 -0.0524  ...  0.0435  0.0270 -0.0931 -0.1552 -0.0092  0.0478 -0.0956 -0.0634     e-mail

[5 rows x 301 columns]
>>> df[df['token'].str.contains('-')].sample(100)['token']
Sun-woo                      Sun-woo
wooden-handled        wooden-handled
quarter-ton              quarter-ton
call-handling          call-handling
mass-deleting          mass-deleting
                          ...       
Church-Turing          Church-Turing
65-yard                      65-yard
bear-hugged              bear-hugged
Simpsons-related    Simpsons-related
Lance-Star                Lance-Star
Name: token, Length: 100, dtype: object
>>> df[df['token'].str.contains('-')].sample(10)['token']
Chestnut-backed    Chestnut-backed
non-welfare            non-welfare
23-mile                    23-mile
NS-2                          NS-2
Trois-Ponts            Trois-Ponts
Embry-Riddle          Embry-Riddle
mud-puddling          mud-puddling
K-32                          K-32
DSR-1                        DSR-1
sub-indices            sub-indices
Name: token, dtype: object
>>> df[df['token'].str.contains('.')].sample(10)['token']
fantasia            fantasia
UNSIGNED            UNSIGNED
--Epeefleche    --Epeefleche
Panchobh            Panchobh
askign                askign
WBCC                    WBCC
Romaniotes        Romaniotes
Đời                      Đời
nipple                nipple
rayo                    rayo
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[.]')].sample(10)['token']
Statoids.com                    Statoids.com
sci.physics.research    sci.physics.research
4.13                                    4.13
27.98                                  27.98
53.96                                  53.96
10.1017                              10.1017
1.35pm                                1.35pm
.215                                    .215
walk.                                  walk.
Ayubowan.png                    Ayubowan.png
Name: token, dtype: object
>>> df[df['token'].str.contains(r'\[')].sample(10)['token']
>>> df[df['token'].str.contains(r'[\[]')].sample(10)['token']
>>> df[df['token'].str.contains(r'[[]')].sample(10)['token']
>>> df[df['token'].str.contains(r'[^]')].sample(10)['token']
>>> df[df['token'].str.contains(r'[\^]')].sample(10)['token']
>>> df[df['token'].str.contains(r'[\^]')].sample()['token']
^    ^
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[\~]')].sample()['token']
~    ~
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[\~]')].sample(3, replace=True)['token']
~    ~
~    ~
~    ~
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[\!]')].sample(3, replace=True)['token']
!    !
!    !
!    !
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[\_]')].sample(3, replace=True)['token']
>>> df[df['token'].str.contains(r'[ ]')].sample(3, replace=True)['token']
>>> df[df['token'].str.contains(r'[|]')].sample(3, replace=True)['token']
|    |
|    |
|    |
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[,]')].sample(3, replace=True)['token']
22,950      22,950
274,000    274,000
4,746        4,746
Name: token, dtype: object
>>> df[df['token'].str.contains(r'[@]')].sample(3, replace=True)['token']
@    @
@    @
@    @
Name: token, dtype: object
>>> hist -o -p -f 'src/nlpia2/ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md'
>>> ls
>>> pwd
'/home/hobs/code/tangibleai/nlpia2/src/nlpia2'
>>> hist -o -p -f 'ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md'
>>> hist -o -p -f ch06/nessvec-load_fasttext-1Mx300D-vocabulary.ipy.md
>>> hist -f ch06/nessvec_load_fasttext_1Mx300D_vocabulary.py
>>> hist -f ch06/nessvec_load_fasttext_1Mx300D_vocabulary.hist.py
>>> hist -o -p -f ch06/nessvec-load_fasttext-1Mx300D-vocabulary.hist.ipy.md
