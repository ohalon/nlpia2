import spacy
import pandas as pd
import numpy as np
pd.options.display.max_columns = 2000

# spacy.cli.download('en_core_web_md')
df = pd.read_csv("../nlpia2/src/nlpia2/data/nlpia_lines.csv.gz")
df3 = df[df["filename"] == "Chapter-03_Math-with-Words-TF-IDF-Vectors.adoc"]
texts = df3.text[df3.is_text | df3.is_title]
nlp = spacy.load("en_core_web_md")
embeddings = texts.apply(lambda s: nlp(s).vector)
dfe = pd.DataFrame([list(x / np.linalg.norm(x)) for x in embeddings])
heatmap = dfe.values.dot(dfe.values.T)
heatmap.shape


import seaborn as sn
import matplotlib.pyplot as plt

hm = sn.heatmap(data=heatmap, vmin=0.6, vmax=1)
plt.tight_layout()
plt.show()

closeones = []
for i1, text1 in enumerate(texts):
    for i2, text2 in enumerate(texts[i1 + 1:]):
        i3 = i2 + i1 + 1
        if heatmap[i1][i3] > 0.95:
            closeones.append(
                dict(
                    coord=[i1, i3],
                    value=heatmap[i1][i3],
                    text1=text1,
                    text1b=df.text.iloc[i1],
                    text2=text2,
                    text2b=df.text.iloc[i3],
                )
            )
dfclose = pd.DataFrame(closeones)
dfclose
dfclose[["coord", "value", "text1", "text2"]].iloc[0].values
