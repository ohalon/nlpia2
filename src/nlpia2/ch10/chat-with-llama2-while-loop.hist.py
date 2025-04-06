from transformers import LlamaForCausalLM, LlamaTokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
llama = LlamaForCausalLM.from_pretrained(
    model_name,
    use_auth_token=auth_token
    )
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    token=auth_token
    )
prompt = "Are you able to reason about things like math word problems or logical inference?"
inputs = tokenizer(prompt, return_tensors="pt")
output_token_ids = llama.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(output_token_ids)[0]
import dotenv
import os
dotenv.load_dotenv()
env = dict(os.environ)
auth_token = env['HUGGINGFACE_ACCESS_TOKEN']
>>> from transformers import LlamaForCausalLM, LlamaTokenizer
>>> model_name = "meta-llama/Llama-2-7b-chat-hf"
auth_token
>>> llama = LlamaForCausalLM.from_pretrained(
...     model_name,
...     token=auth_token)
>>> tokenizer = LlamaTokenizer.from_pretrained(
...     model_name,
...     token=auth_token)
>>> prompt = "Q: How do you know when you misunderstand the real world?\n"
>>> prompt += "A: "
>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
>>> input_ids
input_len = len(input_ids[0])
input_len = input_ids.size[0]
input_len = input_ids.size()[0]
input_len = input_ids.size()
input_ids.size()
input_len = input_ids.size()[1]
tok = ''

while tok.strip() not in ('', '</s>'):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_len = input_ids.size()[0]
    output_ids = llama.generate(
        input_ids, max_length=input_len + 1)
    tok_id = list(output_ids[0])[-1]
    output_str = tokenizer.batch_decode(output_ids)[0]
    print(output_str)
    tok = output_str[len(prompt):] 
    print(tok)
    prompt += tok
    print()

while tok.strip() is not '</s>':
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_len = input_ids.size()[0]
    output_ids = llama.generate(
        input_ids, max_length=input_len + 1)
    tok_id = list(output_ids[0])[-1]
    output_str = tokenizer.batch_decode(output_ids)[0]
    print(output_str)
    tok = output_str[len(prompt):] 
    print(tok)
    prompt += tok
    print()
hist -o -p -f src/nlpia2/ch10/chat-with-llama2-while-loop.hist.ipy
hist -f src/nlpia2/ch10/chat-with-llama2-while-loop.hist.py
