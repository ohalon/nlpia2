import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression, Lasso
from nlpia.constants import DATA_DIR

df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')

# df = df.sample(1_000_000, random_state=1989)
np.random.seed(451)
istrain = np.random.rand(len(df)) < .9

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(df['name'][istrain])
vecs = vectorizer.transform(df['name'])
