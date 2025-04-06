import pandas as pd

url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2'

url += '/data/llm/llm-emmergence-table-other-big-bench-tasks.csv'

df = pd.read_csv(url, index_col=0)

df.shape  # <1>

df['Emergence'].value_counts()

scales = df['Emergence'].apply(lambda x: 'line' in x or 'flat' in x)

df[scales].sort_values('Task')  # <3>

import numpy as np

np.random.choice(
    'statistical,AI,stochastic,interesting,a,an,in,of'.split(','),
    p=[.18, .17, .15, .1, .1, .1, .1, .1])  # <1>

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch

import numpy as np

SEED = 42

DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    DEVICE = torch.cuda.device(0)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED) # <1>

from transformers import set_seed

set_seed(SEED)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token  # <1>

vanilla_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

def generate(prompt,
       model=vanilla_gpt2,
       tokenizer=tokenizer,
       device=DEVICE, **kwargs):

   encoded_prompt = tokenizer.encode(
       prompt, return_tensors='pt')

   encoded_prompt = encoded_prompt.to(device)

   encoded_output = model.generate (encoded_prompt, **kwargs)

   encoded_output = encoded_output.squeeze() # <1>

   decoded_output = tokenizer.decode(encoded_output,
       clean_up_tokenization_spaces=True,
       skip_special_tokens=True)

   return decoded_output

generate(
    model=vanilla_gpt2,
    tokenizer=tokenizer,
    prompt='NLP is',
    max_length=50)

input_ids = tokenizer.encode(prompt, return_tensors="pt")

input_ids = input_ids.to(DEVICE)

vanilla_gpt2(input_ids=input_ids)

output = vanilla_gpt2(input_ids=input_ids)

output.logits.shape

encoded_prompt = tokenizer('NLP is a', return_tensors="pt")

encoded_prompt = encoded_prompt["input_ids"]

encoded_prompt = encoded_prompt.to(DEVICE)

output = vanilla_gpt2(input_ids=encoded_prompt)

next_token_logits = output.logits[0, -1, :]

next_token_probs = torch.softmax(next_token_logits, dim=-1)

sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

tokenizer.decode(sorted_ids[0])  # <1>

tokenizer.decode(sorted_ids[1])  # <2>

kwargs = {
   'do_sample': True,
   'max_length': 50,
   'top_p': 0.92
}

print(generate(prompt='NLP is a', **kwargs))

import pandas as pd

DATASET_URL = ('https://gitlab.com/tangibleai/nlpia2/'
    '-/raw/main/src/nlpia2/data/nlpia_lines.csv')

df = pd.read_csv(DATASET_URL)

df = df[df['is_text']]

lines = df.line_text.copy()

from torch.utils.data import Dataset

from torch.utils.data import random_split

class NLPiADataset(Dataset):

    def __init__(self, txt_list, tokenizer, max_length=768):

        self.tokenizer = tokenizer

        self.input_ids = []

        self.attn_masks = []

        for txt in txt_list:

            encodings_dict = tokenizer(txt, truncation=True,
                max_length=max_length, padding="max_length")

            self.input_ids.append(
                torch.tensor(encodings_dict['input_ids']))

    def __len__(self):

        return len(self.input_ids)

    def __getitem__(self, idx):

        return self.input_ids[idx]

dataset = NLPiADataset(lines, tokenizer, max_length=768)

train_size = int(0.9 * len(dataset))

eval_size = len(dataset) - train_size

train_dataset, eval_dataset = random_split(
    dataset, [train_size, eval_size])

from nlpia2.constants import DATA_DIR  # <1>

from transformers import TrainingArguments

from transformers import DataCollatorForLanguageModeling

training_args = TrainingArguments(
   output_dir=DATA_DIR / 'ch10_checkpoints',
   per_device_train_batch_size=5,
   num_train_epochs=5,
   save_strategy='epoch')

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)  # <2>

from transformers import Trainer

model = GPT2LMHeadModel.from_pretrained("gpt2")  # <1>

trainer = Trainer(
       model,
       training_args,
       data_collator=collator,       # <2>
       train_dataset=train_dataset,  # <3>
       eval_dataset=eval_dataset)

trainer.train()

generate('NLP is')

print(generate("Neural networks", **nucleus_sampling_args))

print(generate("Neural networks", **nucleus_sampling_args))

import numpy as np

v = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555])

type(v[0])

(v * 1_000_000).astype(np.int32)

v = (v * 1_000_000).astype(np.int32)  # <1>

v = (v + v) // 2

v / 1_000_000

v = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555])

v = (v * 10_000).astype(np.int16)  # <1>

v = (v + v) // 2

v / 10_000

v = np.array([1.1, 2.22, 3.333, 4.4444, 5.55555])

v = (v * 1_000).astype(np.int16)  # <3>

v = (v + v) // 2

v / 1_000

import pandas as pd

DATASET_URL = ('https://gitlab.com/tangibleai/nlpia2/'
    '-/raw/main/src/nlpia2/data/nlpia_lines.csv')

df = pd.read_csv(DATASET_URL)

df = df[df['is_text']]

from haystack import Document

titles = list(df["line_text"].values)

texts = list(df["line_text"].values)

documents = []

for title, text in zip(titles, texts):
   documents.append(Document(content=text, meta={"name": title or ""}))

documents[0]

from haystack.document_stores import FAISSDocumentStore

document_store = FAISSDocumentStore(
    return_embedding=True)  # <1>

document_store.write_documents(documents)

from haystack.nodes import TransformersReader, EmbeddingRetriever

reader = TransformersReader(model_name_or_path
    ="deepset/roberta-base-squad2")  # <1>

retriever = EmbeddingRetriever(
   document_store=document_store,
   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

document_store.update_embeddings(retriever=retriever)

document_store.save('nlpia_index_faiss')  # <2>

from haystack.pipelines import Pipeline

pipe = Pipeline()

pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])

pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])

from haystack.pipelines import ExtractiveQAPipeline

pipe= ExtractiveQAPipeline(reader, retriever)

question = "What is an embedding?"

result = pipe.run(query=question,
    params={"Generator": {
        "top_k": 1}, "Retriever": {"top_k": 5}})

print_answers(result, details='minimum')

from haystack.nodes import Seq2SeqGenerator

from haystack.pipelines import GenerativeQAPipeline

generator = Seq2SeqGenerator(
    model_name_or_path="vblagoje/bart_lfqa",
    max_length=200)

pipe = GenerativeQAPipeline(generator, retriever)

question = "How CNNs are different from RNNs"

result = pipe.run( query=question,
       params={"Retriever": {"top_k": 10}})  # <1>

print_answers(result, details='medium')

question = "How can artificial intelligence save the world"

result = pipe.run(
    query="How can artificial intelligence save the world",
    params={"Retriever": {"top_k": 10}})

result

import streamlit as st

st.title("Ask me about NLPiA!")

st.markdown("Welcome to the official Question Answering webapp"
    "for _Natural Language Processing in Action, 2nd Ed_")

question = st.text_input("Enter your question here:")

if question:
   st.write(f"You asked: '{question}'")

def load_store():
  return FAISSDocumentStore.load(index_path="nlpia_faiss_index.faiss",
                                 config_path="nlpia_faiss_index.json")

@st.cache_resource

def load_retriever(_document_store):    #<1>
   return EmbeddingRetriever(
    document_store=_document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
   )

@st.cache_resource

def load_reader():
   return TransformersReader(
       model_name_or_path="deepset/roberta-base-squad2")

document_store = load_store()

extractive_retriever = load_retriever(document_store)

reader = load_reader()

pipe = ExtractiveQAPipeline(reader, extractive_retriever)

if question:
   res = pipe.run(query=question, params={

import nlpia2_wikipedia.wikipedia as wiki

wiki.page("AI")

import nlpia2_wikipedia.wikipedia as wiki

page = wiki.page('AI')

page.title

print(page.content)

wiki.search('AI')

wiki.set_lang('zh')

wiki.search('AI')
