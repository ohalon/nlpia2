import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

dfs = pd.read_html('http://www.sideroad.com/Business_Communication/politically-correct-language.html')
df = dfs[0]
df.columns = 'insensitive sensitive'.split()

texts = [str(s) for s in df['insensitive']]
vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(1, 3), stop_words=None)
vectorizer.fit(texts)
# CountVectorizer(max_df=0.5, ngram_range=(1, 3))
vectorizer.transform(texts)
countvecs = vectorizer.transform(texts).todense()
countvecs
# matrix([[0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         ...,
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0]])
cv = pd.DataFrame(countvecs, columns=vectorizer.get_feature_names())
#     acting  acting blonde  acting like  acting like wild  adults  ...  wild  wild indians  woman  woman the  woman the wife
# 0        0              0            0                 0       0  ...     0             0      0          0               0
# 1        0              0            0                 0       0  ...     0             0      0          0               0
#
# [26 rows x 154 columns]
cv.T
#                   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25
# acting             0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
# acting blonde      0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0
#
# [154 rows x 26 columns]
qv = m[0]
# cv
#     acting  acting blonde  acting like  acting like wild  adults  and  and the  and the theater  ...  white  white lie  wife  wild  wild indians  woman  woman the  woman the wife
# 0        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
# 1        0              0            0                 0       0    0        0                0  ...      0          0     0     0             0      0          0               0
cv.dot(qv)
# 0     0
# 1     0
# 2     0
# 3     2
# 4     0
# 5     0
# 6     0
cv.dot(qv).sort_values(ascending=False)
# 3     2
# 16    1
# 0     0
# 14    0
query = 'wild acting'
m = vectorizer.transform([query]).toarray()
qv = m[0]
cv.dot(qv).sort_values(ascending=False).index[0]
# 3
texts[cv.dot(qv).sort_values(ascending=False).index[0]]
# 'Acting like wild Indians'
vectorizer = CountVectorizer(min_df=1, max_df=.5, ngram_range=(3, 3), stop_words=None, analyzer='char')
# CountVectorizer(analyzer='char', max_df=0.5, ngram_range=(3, 3))
trgcounts = pd.DataFrame(vectorizer.transform(texts).toarray(), columns=vectorizer.get_feature_names())
trgcounts.columns[0]
' "p'
vectorizer.transform([query])
# <1x334 sparse matrix of type '<class 'numpy.int64'>'
#     with 7 stored elements in Compressed Sparse Row format >
qv = vectorizer.transform([query]).toarray()[0]
trgcounts.dot(qv)
# 0     0
# 1     1
# 2     1
# 3     7
# 4     1
query = 'wild acting'
query = 'wld actng'
trgcounts.dot(qv)
# 0     0
# 1     1
# 2     1
# 3     7
# 4     1
# 5     0
qv = vectorizer.transform([query]).toarray()[0]
trgcounts.dot(qv)
# 0     0
# 1     0
# 2     0
# 3     2
# 4     0
# 5     0
query = 'wild actin'
qv = vectorizer.transform([query]).toarray()[0]
trgcounts.dot(qv)
# 0     0
# 1     0
# 2     0
# 3     6
# 4     0
# 5     0
trgcounts.sum()
#  "p    1
#  (w    5
#  a     1
#  ad    1
#  an    1
#       ..
# wor    1
# xed    1
# xis    1
# y o    1
# ys"    1
# Length: 334, dtype: int64
counts = trgcounts.sum()
counts.index
# Index([' "p', ' (w', ' a ', ' ad', ' an', ' ar', ' bl', ' bo', ' ch', ' co',
from collections import Counter
from collections import Counter
c = Counter()
for bg, num in zip(counts.index.str[:2], counts.values):
    c += Counter((bg, num))
c
# Counter({' "': 1,
#          1: 268,
#          ' (': 1,
#          5: 16,
counts.sum()
# 485
prob = pd.Series(c) / counts.sum()
seed='he'
for i in range(30):
    try:
        seed += prob.get(seed[-2:])
    except TypeError:
        break
prob.get('he')
0.010309278350515464
prob
#  "    0.002062
# 1     0.552577
#  (0.002062
# 5     0.032990
# [bigram for bigram, onegram in zip(*c.keys())]
