import streamlit as st
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

def load_store():
    return FAISSDocumentStore.load(index_path="nlpia_faiss_index.faiss",
                                             config_path="nlpia_faiss_index.json")

@st.cache_resource
def load_retriever(_document_store):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )

@st.cache_resource
def load_reader():
    return TransformersReader(model_name_or_path="deepset/roberta-base-squad2")


st.title("Ask me about NLPiA!")
st.markdown("Welcome to the official web app of Natural Language Processing in Action, 2nd edition.")

document_store = load_store()
extractive_retriever = load_retriever(document_store)
reader = load_reader()
pipe = ExtractiveQAPipeline(reader, extractive_retriever)

question = st.text_input("Enter your question here 💭 ")

if question:
    res = pipe.run(query=question, params={"Reader": {"top_k": 1}, "Retriever": {"top_k": 10}})
    st.write(f"Answer: {res['answers'][0].answer}")
    st.write(f"Context: {res['answers'][0].context}")




### Beginning of sandbox with expanders 
# list_len = 3
# labels = ["label1", "label2", "label3"]
# texts = ["text1", "text2", "text3"]

# expanders = [0]*list_len
# for i in range(list_len):
