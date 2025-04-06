from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed
from nlpia2.constants import ENV
"""
prompt = "How do you (Llama2) know when you make common sense errors?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(prompt, end='', flush=True)
while not prompt.endswith('</s>'):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_len = len(input_ids[0])
    output_ids = llama.generate(
        input_ids, max_length=input_len + 1)
    ans_ids = output_ids[0][input_len:]
    output_str = tokenizer.batch_decode(
        output_ids, skip_special_tokens=False)[0]
    output_str = output_str[3:]  # <1>
    tok = output_str[len(prompt):]
    print(tok, end='', flush=True)
    prompt = output_str
"""


DEFAULT_LLM = "lmsys/vicuna-13b-v1.5"
# ENV.get("DEFAULT_LLM", "meta-llama/Llama-2-7b-chat-hf")
HF_TOKEN = ENV.get("HUGGINGFACE_ACCESS_TOKEN", "hf_...")


class Bot:

    def __init__(self, auth_token=HF_TOKEN, model_name=DEFAULT_LLM, prompt="", random_seed=0):
        self.random_seed = random_seed
        self.message_log = []
        self.auth_token = auth_token
        set_seed(random_seed)
        self.auth_token = ENV['HUGGINGFACE_ACCESS_TOKEN']
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            token=self.auth_token)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            use_auth_token=self.auth_token
        )

    def send_prompt(self, prompt='What do you think of ChatGPT?\n'):
        # print(prompt, end='', flush=True)
        while not prompt.endswith('</s>') and not prompt.strip().lower().endswith('human:'):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_len = len(input_ids[0])
            output_ids = self.model.generate(
                input_ids, max_length=input_len + 1)
            # ans_ids = output_ids[0][input_len:]
            output_str = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=False)[0]
            output_str = output_str[3:]  # <1>
            tok = output_str[len(prompt):]
            print(tok, end='', flush=True)
            prompt = output_str
        self.log(prompt)
        return prompt

    def log(self, message=None):
        if message is not None:
            self.message_log.append(message)
        return self.message_log

    def chat(self,
             prompt=('You are a smart and concise AI assistant. '
                     'Complete the following dialog in one sentence or less.\n'),
             human_prefix='Human:',
             ai_prefix="AI Assistant:"):
        print(prompt, end='', flush=True)
        while prompt and prompt.lower().strip() != 'exit':
            prompt += '\n' + human_prefix + input(f'{human_prefix}: ')
            ai_prompt = ai_prefix
            print(ai_prompt, end='', flush=True)
            prompt += f'\n{ai_prompt}'
            self.log(self.send_prompt(prompt=prompt))


if __name__ == '__main__':
    cli = Bot()
    cli.chat()
