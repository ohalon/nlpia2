


"""## Logging

We configure how logging messages should be displayed and which log level should be used before importing Haystack.
Example log message:
INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
"""

import pandas as pd

# TODOHL: CH10 - Create dataframe with columns "title" and "text"
url = (
    'https://gitlab.com/tangibleai/nlpia2/-/raw/main/'
    'src/nlpia2/data/nlpia_lines.csv')
df = pd.read_csv(url, index_col=0)
df = df[df['is_text']]
df['title'] = df['line_text']
df['text'] = df['line_text']

print(df.head())

"""We can cast our data into Haystack Document objects.
Alternatively, we can also just use dictionaries with "text" and "meta" fields
"""

from haystack import Document


# Use data to initialize Document objects
titles = list(df["title"].values)
texts = list(df["text"].values)
documents = []
for title, text in zip(titles, texts):
    documents.append(Document(content=text, meta={"name": title or ""}))
documents[0]

"""Here we initialize the FAISSDocumentStore, DensePassageRetriever and RAGenerator.
FAISS is chosen here since it is optimized vector storage.
"""

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, DensePassageRetriever

# TODOHL: CH10 - create a data_dir to store the index and write a config.json file to it

import json
from nlpia2.constants import BIGDATA_DIR
faiss_dir = BIGDATA_DIR / 'faiss'
faiss_dir.mkdir(exist_ok=True, parents=True)
index_path = faiss_dir / "nlpia.faiss"
config_path = faiss_dir / "nlpia.faiss.json"
config = {
    "faiss_index_factory_str": "HNSW",
    "return_embedding": True
    }
json.dump(config, config_path.open('w'))

# Initialize FAISS document store.
# Set `return_embedding` to `True`, so generator doesn't have to perform re-embedding
# document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW", return_embedding=True)

# first time
docstore = FAISSDocumentStore(
    embedding_dim=128,
    faiss_index_factory_str="HNSW",
    return_embedding=True)

# Initialize DPR Retriever to encode documents, encode question and query documents
retriever = DensePassageRetriever(
    document_store=docstore,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
  #  use_gpu=True,
    embed_title=True,
)

# Initialize RAG Generator
generator = RAGenerator(  # <1> this downloads a pretrained .5GB model & 2GB BART tokenizer!
    model_name_or_path="facebook/rag-token-nq",
    # use_gpu=True,
    top_k=1,
    max_length=15,
    min_length=5,
    embed_title=True,
    num_beams=3,
)

"""We write documents to the DocumentStore, first by deleting any remaining documents then calling `write_documents()`.
The `update_embeddings()` method uses the retriever to create an embedding for each document.

"""

# Delete existing documents in documents store
docstore.delete_documents()

# Write documents to document store
docstore.write_documents(documents)

# Add documents embeddings to index
docstore.update_embeddings(retriever=retriever)

"""Here are our questions:"""

QUESTIONS = [
  "how Word2vec compares to LSA", 
  "what is an intent", 
  "what is an embedding", 
  "who discovered LDiA", 
  "what are the most advanced NLP models", 
  "what are RNNs good for"
]

"""Now let's run our system!
The retriever will pick out a small subset of documents that it finds relevant.
These are used to condition the generator as it generates the answer.
What it should return then are novel text spans that form and answer to your question!
"""

# Or alternatively use the Pipeline class
from haystack.pipelines import GenerativeQAPipeline
from haystack.utils import print_answers

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
    print(res)
    query = res['query']
    answer = res['answers'][0].answer
    context = ('\n').join(res['answers'][0].meta['content'])
    print(f'Query:{query}')
    print(f'Answer:{answer}')
    print(f'Context:{context}\n\n')

from haystack.nodes import EmbeddingRetriever
extractive_retriever = EmbeddingRetriever(
    document_store = document_store, 
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

# Add documents embeddings to index
document_store.update_embeddings(retriever=extractive_retriever)

pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
for question in QUESTIONS:
    res = pipe.run(query=question, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 10}})
    print_answers(res, details='medium')

"""## Extractive QA with Haystack"""

from haystack.nodes import EmbeddingRetriever

document_store.delete_documents()

# Write documents to document store
document_store.write_documents(documents)

extractive_retriever = EmbeddingRetriever(
    document_store = document_store, 
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

document_store.update_embeddings(retriever=extractive_retriever)

from haystack.nodes import TransformersReader
reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2")

from haystack.pipelines import ExtractiveQAPipeline
p = ExtractiveQAPipeline(reader, extractive_retriever)
for question in QUESTIONS:
    res = p.run(query=question, params={"Reader": {"top_k": 1}, "Retriever": {"top_k": 10}})
    print_answers(res, details='minimum')

def load_pipeline():
    document_store = FAISSDocumentStore.load(index_path="nlpia_faiss_index.faiss",
                                             config_path="nlpia_faiss_index.json")

    extractive_retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )

    reader = TransformersReader(model_name_or_path="deepset/roberta-base-squad2")

    p = ExtractiveQAPipeline(reader, extractive_retriever)
    return p
