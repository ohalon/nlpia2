from txtai.embeddings import Embeddings
import pandas as pd
from nlpia2.constants import SRC_DATA_DIR

embeddings = Embeddings(
    {"path": "sentence-transformers/nli-mpnet-base-v2"}
    )


df = pd.read_csv(SRC_DATA_DIR / 'nlpia_lines.csv.gz')
print("%-20s %s" % ("Query", "Best Match"))
print("-" * 50)

for query in ("Word embeddings are good for semantic search of words or dictionaries and synonyms.", ):
    # Get index of best section that best matches query
    uid = embeddings.similarity(query, df['text'])[0][0]

    print("%-20s %s" % (query, df['text'][uid]))