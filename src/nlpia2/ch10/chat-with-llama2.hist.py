""" Download and run Llama2 locally. WARNING: You need at least 48GB of RAM for this to work

See: https://huggingface.co/docs/transformers/v4.33.2/en/model_doc/llama2#transformers.LlamaForCausalLM.forward.example

"""
from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM, pipeline
import dotenv
import os
dotenv.load_dotenv()
env = dict(os.environ)
auth_token = env['HUGGINGFACE_ACCESS_TOKEN']
model_name = "meta-llama/Llama-2-7b-chat-hf"
llama = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=auth_token)
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=auth_token)
prompt = "Are you able to reason about things like math word problems or logical inference?"
inputs = tokenizer(prompt, return_tensors="pt")
output_token_ids = llama.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(output_token_ids)[0]
