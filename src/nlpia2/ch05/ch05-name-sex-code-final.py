>>> import pandas as pd
>>> import numpy as np
>>> pd.options.display.max_rows = 7

>>> np.random.seed(451)
>>> df = pd.read_csv('https://proai.org/baby-names-us.csv.gz')  # <1>
>>> df = df.sample(10_000)
>>> df

>>> df = df.set_index(['name', 'sex'])
>>> groups = df.groupby(['name', 'sex'])
>>> counts = groups['count'].sum()
>>> counts