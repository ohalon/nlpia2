# import numpy as np
import pandas as pd
# import re
from pathlib import Path

from sklearn.linear_model import LogisticRegression
# from pandas.api.types import is_numeric_dtype

import seaborn as sns
from matplotlib import pyplot as plt
# import pandas as pd

sns.set_theme(color_codes=True)
plt.rcParams['figure.figsize'] = [6, 4.5]
plt.rcParams['figure.dpi'] = 120

# THIS_DIR = Path(__file__).expanduser().resolve().absolute().parent
# DATA_DIR = Path(THIS_DIR).parent.parent / '.nlpia2-data'
DATA_DIR = Path('/home/hobs/code/prosocialai/nlpia2/.nlpia2-data')
from nlpia2.constants import MANUSCRIPT_DIR
CH05_IMAGES_DIR = MANUSCRIPT_DIR / 'manuscript' / 'images' / 'ch05'
DATAFILE = DATA_DIR / 'fake_vs_real.csv.gz'  # 'all.csv.gz'
assert CH05_IMAGES_DIR.is_dir()

df = pd.read_csv(DATAFILE, index_col=None)
df['title_len'] = df['title'].str.len()
# feature_names = ['title_len', 'text_len', 'title_allcap_ratio', 'title_allcap_token_len']
feature_names = ['title_len']
target_name = 'isfake'

X = df[feature_names]
y = df[target_name]

model = LogisticRegression(class_weight='balanced')
model.fit(X[feature_names], y)
print(model.score(X[feature_names], y))


plt.figure()
ax = sns.regplot(data=df.sample(200), x=feature_names[0], y=target_name,
                 logistic=True)
plt.xlabel('Title Length (characters)')
plt.ylabel('Fake News')
print(Path(__file__).name)
image_filepath = CH05_IMAGES_DIR / (Path(__file__).name[:-2] + 'png')
print(f'Saving logistic regression plot to {image_filepath}')
plt.savefig(image_filepath)
plt.show()
