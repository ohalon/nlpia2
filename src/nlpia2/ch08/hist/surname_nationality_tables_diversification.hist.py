%run char_rnn_from_scratch_refactored
df = load_names_from_text(dedupe=True, categories=None)
df = load_names_from_text(dedupe=True, categories=None).sample(10)
df = load_names_from_text(dedupe=True, categories=None)
len(df.sample(100))
len(df.groupby('category').sample(100))
len(df.groupby('category').sample(100, replace=True))
df.sample?
len(df.groupby('category').sample(frac=1.0, replace=True))
len(df)
categories
CATEGORIES
CATEGORIES == df['category'].unique()
[c in CATEGORIES for c in df['category'].unique()]
len(CATEGORIES)
pd.read_html('https://forebears.io/ethiopia/surnames')
s = """
Rank    Surname Incidence   Frequency
1   Tesfaye 1,167,260   1:84
2   Mohammed    1,084,839   1:90
3   Getachew    895,366 1:109
4   Abebe   825,501 1:118
5   Girma   822,765 1:119
6   Tadesse 816,808 1:119
7   Solomon 672,249 1:145
8   Kebede  660,498 1:148
9   Bekele  635,868 1:153
10  Hailu   585,964 1:166
11  Alemayehu   585,159 1:167
12  Ahmed   534,934 1:182
13  Alemu   526,724 1:185
14  Almaz   520,124 1:188
15  Mulu    510,465 1:191
16  Teshome 503,221 1:194
17  Mekonnen    502,255 1:194
18  Genet   486,479 1:201
19  Abera   479,879 1:203
20  Mulugeta    455,571 1:214
21  Tilahun 400,838 1:243
22  Worku   390,536 1:250
23  Tsegaye 390,214 1:250
24  Ali 383,452 1:254
25  Tsehay  381,843 1:255
26  Asefa   376,208 1:259
27  Abebech 368,803 1:264
28  Jemal   364,618 1:268
29  Assefa  343,369 1:284
30  Desta   339,827 1:287
31  Birhanu 339,505 1:287
32  Mesfin  335,964 1:290
33  Yeshi   329,524 1:296
34  Meseret 327,915 1:297
35  Kedir   326,144 1:299
36  Seid    325,339 1:300
37  Mohamed 320,027 1:305
38  Sisay   312,783 1:312
39  Berhanu 299,743 1:325
40  Belay   293,626 1:332
41  Eshetu  287,187 1:340
42  Aster   283,967 1:344
43  Ayele   282,679 1:345
44  Tefera  276,401 1:353
45  Haile   273,665 1:356
46  Ayalew  256,118 1:381
47  Tigist  251,289 1:388
48  Dereje  250,162 1:390
49  Belaynesh   248,874 1:392
50  Fatuma  247,425 1:394
51  Zenebech    244,044 1:400
52  Getahun 241,952 1:403
53  Amare   238,249 1:409
54  Hassen  234,869 1:415
55  Mengistu    231,166 1:422
56  Abdi    230,522 1:423
57  Alem    228,269 1:427
58  Negash  223,761 1:436
59  Abeba   221,507 1:440
60  Hussen  219,576 1:444
61  Desalegn    214,746 1:454
62  Shiferaw    212,815 1:458
63  Taye    211,688 1:461
64  Kassa   210,400 1:464
65  Asfaw   205,731 1:474
66  Emebet  205,248 1:475
67  Belete  204,927 1:476
68  Mamo    201,385 1:484
69  Tsige   199,936 1:488
70  Beyene  198,487 1:491
71  Alemitu 195,268 1:500
72  Asnakech    194,624 1:501
73  Etenesh 193,336 1:505
73  Fekadu  193,336 1:505
75  Aregash 191,243 1:510
76  Aberash 185,931 1:525
77  Askale  185,126 1:527
78  Abdela  183,999 1:530
79  Melaku  183,677 1:531
80  Dawit   179,814 1:542
81  Bizunesh    178,204 1:547
82  Yohannes    176,433 1:553
83  Atsede  174,984 1:557
84  Abate   174,501 1:559
85  Asrat   173,375 1:563
86  Temesgen    170,960 1:571
87  Ibrahim 170,477 1:572
88  Getu    168,062 1:580
89  Habtamu 167,096 1:584
90  Fikadu  165,648 1:589
91  Moges   164,038 1:595
92  Dejene  163,233 1:598
93  Melese  162,589 1:600
94  Adem    160,979 1:606
95  Aynalem 160,174 1:609
96  Lemma   159,369 1:612
97  Ayelech 158,565 1:615
98  Zerihun 152,286 1:641
99  Legesse 149,228 1:654
100 Asegedech   148,262 1:658
101 Mulunesh    147,457 1:662
102 Bogale  147,135 1:663
102 Shewaye 147,135 1:663
104 Adane   146,491 1:666
104 Adanech 146,491 1:666
106 Tadele  145,847 1:669
107 Berhane 145,364 1:671
108 Abreham 144,237 1:676
109 Endale  143,754 1:679
110 Seyoum  143,111 1:682
111 Ketema  137,959 1:707
112 Molla   137,154 1:711
113 Nigussie    136,832 1:713
114 Kassahun    136,510 1:715
115 Adugna  136,349 1:715
116 Bekelech    135,705 1:719
117 Gashaw  134,257 1:727
118 Aselefech   133,774 1:729
119 Zeleke  133,613 1:730
120 Samuel  133,452 1:731
121 Zelalem 132,808 1:734
122 Zewdu   132,486 1:736
123 Ashenafi    132,325 1:737
124 Tariku  132,003 1:739
125 Zewdie  130,554 1:747
126 Alemnesh    129,105 1:756
127 Abdu    127,496 1:765
128 Fantaye 126,047 1:774
129 Tadelech    125,081 1:780
129 Tewabech    125,081 1:780
131 Terefe  123,793 1:788
132 Mekonen 122,505 1:796
133 Gezahegn    121,056 1:806
134 Wondimu 118,320 1:824
135 Sintayehu   117,837 1:828
136 Demissie    117,676 1:829
137 Hagos   117,354 1:831
138 Tamiru  115,905 1:842
139 Addis   115,583 1:844
139 Kemal   115,583 1:844
141 Ahimed  115,422 1:845
142 Feleke  114,778 1:850
143 Debebe  114,295 1:853
144 Lema    113,329 1:861
145 Mengesha    113,007 1:863
146 Demeke  112,524 1:867
147 Birtukan    111,559 1:874
148 Gizaw   111,237 1:877
148 Yilma   111,237 1:877
148 Yimer   111,237 1:877
151 Worknesh    110,915 1:879
152 Hailemariam 109,788 1:888
153 Birhane 109,466 1:891
154 Teferi  108,983 1:895
155 Gete    108,339 1:900
156 Hussien 108,178 1:902
157 Lemlem  107,856 1:904
158 Kiros   107,051 1:911
159 Tirunesh    106,568 1:915
160 Amina   105,602 1:924
161 Admasu  104,636 1:932
162 Kasahun 103,832 1:939
163 Meaza   103,671 1:941
163 Tiruwork    103,671 1:941
165 Mulatu  103,349 1:944
166 Mehammed    103,188 1:945
167 Kidane  102,705 1:950
168 Elias   101,256 1:963
169 Kifle   100,451 1:971
170 Hawa    99,807  1:977
170 Zewde   99,807  1:977
172 Kebebush    99,646  1:979
172 Mitiku  99,646  1:979
174 Kedija  99,163  1:984
175 Takele  98,680  1:989
176 Amelework   97,553  1:1,000
176 Gebeyehu    97,553  1:1,000
176 Hirut   97,553  1:1,000
179 Fantu   96,427  1:1,012
180 Tekle   95,300  1:1,024
181 Berhe   95,139  1:1,025
181 Geremew 95,139  1:1,025
183 Tamirat 94,656  1:1,031
183 Wolde   94,656  1:1,031
185 Alemtsehay  94,495  1:1,032
186 Gebre   94,334  1:1,034
187 Zewditu 93,851  1:1,039
188 Beletu  93,690  1:1,041
189 Mehamed 93,368  1:1,045
190 Nega    91,919  1:1,061
190 Sultan  91,919  1:1,061
192 Teklu   91,758  1:1,063
193 Aman    91,597  1:1,065
193 Gemechu 91,597  1:1,065
193 Tsehaynesh  91,597  1:1,065
196 Tsehaye 90,953  1:1,072
197 Amarech 90,792  1:1,074
198 Abebaw  89,826  1:1,086
198 Zenebe  89,826  1:1,086
200 Belachew    89,343  1:1,092
201 Ebrahim 88,056  1:1,108
202 Aklilu  87,573  1:1,114
202 Elfinesh    87,573  1:1,114
204 Abaynesh    86,768  1:1,124
205 Teka    85,641  1:1,139
206 Hiwot   85,158  1:1,145
206 Niguse  85,158  1:1,145
208 Endris  84,031  1:1,161
209 Selamawit   83,709  1:1,165
210 Legese  83,548  1:1,168
211 Senait  83,387  1:1,170
212 Belayneh    82,904  1:1,177
213 Nasir   82,743  1:1,179
214 Seifu   81,938  1:1,190
215 Yasin   81,777  1:1,193
216 Nigatu  81,134  1:1,202
217 Bogalech    80,651  1:1,209
217 Tibebu  80,651  1:1,209
219 Enanu   79,685  1:1,224
220 Tesfa   79,041  1:1,234
221 Tsedale 78,558  1:1,242
222 Yesuf   78,236  1:1,247
223 Awol    78,075  1:1,249
224 Felekech    77,431  1:1,260
225 Yonas   76,787  1:1,270
226 Husen   76,626  1:1,273
227 Arega   76,304  1:1,278
227 Getaneh 76,304  1:1,278
227 Mekuria 76,304  1:1,278
230 Tesema  75,982  1:1,284
230 Yimam   75,982  1:1,284
232 Girmay  74,855  1:1,303
233 Gidey   73,728  1:1,323
234 Gebremedhin 73,246  1:1,332
235 Tsega   73,085  1:1,335
236 Getnet  72,763  1:1,341
236 Shitaye 72,763  1:1,341
238 Nuru    72,602  1:1,344
239 Mustefa 72,119  1:1,353
240 Tewodros    71,314  1:1,368
241 Birke   71,153  1:1,371
242 Afework 70,992  1:1,374
242 Tarekegn    70,992  1:1,374
244 Azeb    70,831  1:1,377
244 Gebru   70,831  1:1,377
246 Melkamu 70,670  1:1,380
247 Anteneh 70,348  1:1,387
247 Oumer   70,348  1:1,387
247 Tamene  70,348  1:1,387
250 Kelemua 70,026  1:1,393
250 Lakech  70,026  1:1,393
252 Fantahun    69,543  1:1,403
253 Mehari  69,221  1:1,409
254 Yohanes 68,899  1:1,416
254 Zehara  68,899  1:1,416
256 Meselech    68,255  1:1,429
257 Addisu  67,611  1:1,443
258 Mulat   67,450  1:1,446
259 Alganesh    67,289  1:1,450
260 Gizachew    66,806  1:1,460
260 Menbere 66,806  1:1,460
262 Woinshet    66,645  1:1,464
263 Zemzem  66,323  1:1,471
264 Fiseha  66,162  1:1,474
265 Ejigayehu   66,001  1:1,478
266 Kahsay  65,197  1:1,496
267 Yemane  65,036  1:1,500
268 Etaferahu   64,875  1:1,504
269 Markos  64,392  1:1,515
270 Abay    64,231  1:1,519
270 Amsale  64,231  1:1,519
270 Fikre   64,231  1:1,519
273 Abdulahi    63,909  1:1,526
274 Lubaba  63,587  1:1,534
275 Meskerem    63,426  1:1,538
276 Hasen   62,943  1:1,550
277 Mesele  62,782  1:1,554
278 Amsalu  62,460  1:1,562
279 Alebachew   62,138  1:1,570
280 Workinesh   61,816  1:1,578
281 Etagegn 61,333  1:1,590
281 Habte   61,333  1:1,590
283 Kassaye 61,172  1:1,595
284 Kasech  60,528  1:1,612
285 Yosef   60,045  1:1,625
286 Muluneh 59,884  1:1,629
287 Negussie    59,562  1:1,638
287 Tessema 59,562  1:1,638
289 Shimelis    59,401  1:1,642
290 Sara    59,240  1:1,647
291 Dagne   59,079  1:1,651
291 Tesfay  59,079  1:1,651
293 Abrehet 58,918  1:1,656
293 Rahel   58,918  1:1,656
295 Hana    58,757  1:1,660
296 Yared   58,596  1:1,665
297 Adamu   58,435  1:1,669
297 Masresha    58,435  1:1,669
299 Samson  58,113  1:1,679
300 Fikru   57,953  1:1,683
301 Abdulkadir  57,309  1:1,702
302 Bahiru  56,826  1:1,717
302 Marta   56,826  1:1,717
304 Workneh 56,504  1:1,726
305 Worke   56,343  1:1,731
306 Mebrat  56,182  1:1,736
307 Birhan  55,860  1:1,746
307 Zinash  55,860  1:1,746
309 Hussein 55,699  1:1,751
309 Tenagne 55,699  1:1,751
311 Adisu   55,377  1:1,761
311 Eshete  55,377  1:1,761
311 Nuredin 55,377  1:1,761
314 Tsegay  55,216  1:1,767
315 Aminat  54,733  1:1,782
315 Chala   54,733  1:1,782
317 Demekech    54,572  1:1,787
317 Haji    54,572  1:1,787
317 Muhammed    54,572  1:1,787
320 Ahemed  54,411  1:1,793
321 Shemsu  54,250  1:1,798
322 Lakew   54,089  1:1,803
323 Yoseph  53,767  1:1,814
324 Gebremariam 53,606  1:1,820
325 Sileshi 53,445  1:1,825
326 Degu    53,284  1:1,831
327 Zegeye  52,962  1:1,842
328 Halima  52,801  1:1,847
"""
!curl -O https://forebears.io/ethiopia/surnames
more surnames
pd.read_html('/home/hobs/Downloads/surnames/Most Common Ethiopian Surnames & Meanings.html')
surn = [] ; surn.append(pd.read_html('/home/hobs/Downloads/surnames/Most Common Ethiopian Surnames & Meanings.html')[0])
surn = [] ; surn.append(pd.read_html('/home/hobs/Downloads/surnames/Most Common Malaysian Surnames & Meanings.html')[0])
surn = [];
for name in 'Malaysian Ethiopian Nigerian'.split():
    sn = pd.read_html(f"/home/hobs/Downloads/surnames/Most Common {name} Surnames & Meanings.html")[0]
    sn['category'] = name
    surn.append(sn)
urls = ['https://forebears.io/papua-new-guinea/surnames', 'https://forebears.io/malaysia/surnames', 'https://forebears.io/nigeria/surnames'] ; urls2 = ['https://forebears.io/papua-new-guinea#surnames']
who
from pathlib import Path
Path.home / 'hobs'
Path.home() / 'hobs' / 'Downloads' / 'surnames'
data_dir = _
filepaths = data_dir.glob('Most Common *.html'))
filepaths = list(data_dir.glob('Most Common *.html'))
filepaths
data_dir.isfile()
data_dir.is_file()
data_dir.is_dir()
data_dir
data_dir = Path.home() / 'Downloads' / 'surnames'
filepaths = list(data_dir.glob('Most Common *.html'))
filepaths
dfs = []
for fp in filepaths:
    dfs.extend(pd.read_html(fp))
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    dfs.extend(pd.read_html(str(fp)))
df = pd.concat(dfs)
df
len(df)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('& Meanings', '').strip()
    fp = str(fp)
    
    dfs.extend(pd.read_html(str(fp)))
    dfs[-1]['country'] = country
df = pd.concat(dfs)
df
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    
    dfs.extend(pd.read_html(str(fp)))
    dfs[-1]['country'] = country
df = pd.concat(dfs)
df
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank')
    df.columns = 'surname count frequency'.split()
    df['country'] = country
    df.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank', axis=1)
    df.columns = 'surname count frequency'.split()
    df['country'] = country
    df.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank', axis=0)
    df.columns = 'surname count frequency'.split()
    df['country'] = country
    df.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    # df = df.drop('Rank', axis=0)
    df.columns = 'rank surname count frequency'.split()
    df['country'] = country
    df.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank', axis=1)
    df.columns = 'rank surname count frequency'.split()
    df['country'] = country
    dfs.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    country = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank', axis=1)
    df.columns = 'surname count frequency'.split()
    df['country'] = country
    dfs.append(df)
df = pd.concat(dfs)
df
dfs = []
for fp in filepaths:
    nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df = df.drop('Rank', axis=1)
    df.columns = 'surname count frequency'.split()
    df['nationality'] = nationality
    dfs.append(df)
df = pd.concat(dfs)
hist
hist -o -p -f surname_nationality_tables_diversification.hist.md
hist -f surname_nationality_tables_diversification.hist.py
diversity_df = df.copy()
%run char_rnn_from_scratch_refactored
df = load_names_from_text(dedupe=True, categories=None)
df_diversity.columns
diversity_df.columns
df.columns
df.columns = 'surname nationality count'.split()
dftot = pd.concat([df, diversity_df])
len(dftot)
dftot.shape
diversity_df.shape
df.shape
dftot.sample(100)
df = diversity_df.copy()
df.groupby('nationality')['count'].sum()
groups = df.groupby('nationality')
dfs
len(dfs)
df = dfs[-1]
df['normalized_count'] = df['count'] / df['count'].sum()
df
df['normalized_freq'] = df['frequency'].str.split(':')[0].astype(float) / df['frequency'].str.split(':')[1].astype(float)
df['normalized_freq'] = df['frequency'].str.split(':').apply(lambda x: float(x[0])) / df['frequency'].str.split(':').apply(lambda x: float(x[1]))
df['numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
df['denominator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
df['denominator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[1]))
df['normalized_freq'] = df['numerator'] / df['denominator']
df
df[[c for c in df.columns if c.startswith('no')]]
dfs = []
for fp in filepaths:
    nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]

    df.columns = 'rank surname count frequency'.split()
    df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
    df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(x: lambda x: float(x[1]))
    df['nationality'] = nationality
    dfs.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df.columns = 'rank surname count frequency'.split()
    df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
    df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(x: lambda x: float(x[1]))
    df['nationality'] = nationality
    dfs.append(df)
df = pd.concat(dfs)
dfs = []
for fp in filepaths:
    nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df.columns = 'rank surname count frequency'.split()
    df['freq_numerator'] = df['frequency'].str.replace(',','').str.split(':').apply(lambda x: float(x[0]))
    df['freq_denominator'] = df['frequency'].str.replace(',','_').str.split(':').apply(lambda x: float(x[1]))
    df['nationality'] = nationality
    dfs.append(df)
df = pd.concat(dfs)
df
dfs = []
for fp in filepaths:
    nationality = fp.with_suffix('').name.replace('Most Common', '').replace('Surnames & Meanings', '').strip()
    fp = str(fp)
    df = pd.read_html(str(fp))[-1]
    df.columns = 'rank surname count frequency'.split()
    df['frequency'] = df['frequency'].str.replace(',','')
    df['freq_numerator'] = df['frequency'].str.split(':').apply(lambda x: float(x[0]))
    df['freq_denominator'] = df['frequency'].str.split(':').apply(lambda x: float(x[1]))
    df['nationality'] = nationality
    dfs.append(df)
df = pd.concat(dfs)
df
hist -f surname_nationality_tables_diversification.hist.py
hist -o -p -f surname_nationality_tables_diversification.hist.md
diversity_df = df
df = load_names_from_text(dedupe=True, categories=None)
diversity_df
dftot = pd.concat([df, diversity_df])
dftot
diversity_df
df.columns
df.columns = 'surname nationality count'.split()
dftot = pd.concat([df, diversity_df])
dftot
df.to_csv('../../data/names/surnname_nationality_counts.csv.gz', compression='gzip', index=False)
hist -o -p -f surname_nationality_tables_diversification.hist.md
hist -f surname_nationality_tables_diversification.hist.py
