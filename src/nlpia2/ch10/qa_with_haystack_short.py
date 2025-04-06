from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
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

def load_retriever(store):
    ret

pipe = load_pipeline()
question = "What are RNNs good for"
res = pipe.run(query=question, params={"Reader": {"top_k": 1}, "Retriever": {"top_k": 10}})
print(f"Answer: {res['answers'][0].answer}")

extractive_retriever.save('retriever.haystack')

