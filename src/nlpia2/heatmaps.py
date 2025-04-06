from numpy.linalg import norm
import pandas as pd
from sentence_transformers import SentenceTransformer
import seaborn as sns
from matplotlib import pyplot as plt

url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/'
url += 'src/nlpia2/data/nlpia_lines.csv'  # <1>
df = pd.read_csv(url, index_col=0)


def chapters(df=df, text_col='text', chapters=None, show=False, block=False, exponent=2, min_value=0.5, max_value=1.0):
    """ The default plot of 4 chapters (3000 lines) grinds computer to halt even with 64GB RAM """
    dfch = df
    if chapters is not None and 'chapter' in df.columns:
        chapters = [int(chapters)] if isinstance(chapters, (int, str, float)) else [int(c) for c in chapters]
        in_chapters = pd.Series(False, index=df.index)
        for ch in chapters:
            in_chapters |= df['chapter'] == ch
        dfch = dfch[in_chapters]
    texts = dfch[text_col][dfch.is_body]
    minilm = SentenceTransformer('all-MiniLM-L12-v2')
    vecs = minilm.encode(list(texts))
    dfe = pd.DataFrame([list(v / norm(v)) for v in vecs])
    cos_sim = dfe.values.dot(dfe.values.T)
    cos_sim[cos_sim < min_value] = 0
    cos_sim[cos_sim > max_value] = max_value
    cos_sim = cos_sim ** exponent
    labels = [f'{t:10s}:{c:02d}:{i:04d}'
              for (i, c, t)
              in zip(cos_sim['lineno'], cos_sim['chapter'], texts.str[:14])]
    cos_sim = pd.DataFrame(cos_sim, columns=labels, index=labels)
    if not show:
        return cos_sim
        fig = sns.heatmap(cos_sim)
        plt.xticks(rotation=30, ha='right', va='top')
        plt.yticks(rotation=30, ha='right', va='bottom')
        plt.show(block=block)
    return dict(cos_sim=cos_sim, fig=fig)


def graph(**kwargs):
    cos_sim = chapters(**kwargs)
