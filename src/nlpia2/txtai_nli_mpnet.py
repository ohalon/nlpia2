from txtai.embeddings import Embeddings


def get_embeddings(name="nli-mpnet-base-v2"):
    """ sentence-transformers embeddigns wrapped in a textai model """
    if '/' not in name:
        name = 'sentence-transformers/{name}'
    # Create embeddings model, backed by sentence-transformers & transformers
    return Embeddings({"path": name})


def get_mini_embeddings(name='all-MiniLM-L6-v2'):
    return get_embeddings(name=name)