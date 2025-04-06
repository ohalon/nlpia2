import pandas as pd
import numpy as np

import qary
import qary.etl.faqs

df_dict = qary.etl.faqs.load(domains='data')
df = pd.DataFrame(np.array([df_dict['questions'], df_dict['answers']]).T)
df.columns = 'question answer'.split()
df = df[df.question.str.endswith('?')].copy()

df.to_csv('all-faqs.csv', index=False)
