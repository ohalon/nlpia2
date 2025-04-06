from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed
from nlpia2.constants import ENV
""" Command line chat interface for Llama-2-7B"""


DEFAULT_LLM = ENV.get("DEFAULT_LLM", "meta-llama/Llama-2-7b-chat-hf")
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

    def send_prompt(self,
                    prompt='What do you think of ChatGPT?\n',
                    human_prefix='Human:'):
        # print(prompt, end='', flush=True)
        while not prompt.endswith('</s>') and not prompt.strip().lower().endswith(human_prefix.strip().lower()):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_len = len(input_ids[0])
            output_ids = self.model.generate(
                input_ids, max_length=input_len + 1)
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
             system_prompt=(
                 'You are a smart and concise AI Assistant. '
                 + 'Complete the following dialog in one sentence or less.\n'),
             human_prefix='Human:',
             ai_prefix="AI Assistant:"):
        prompt = system_prompt
        while True:
            prompt += '\n' + human_prefix + input(f'{human_prefix}: ')
            print(ai_prefix, end='', flush=True)
            prompt += f'\n{ai_prefix}'
            self.log(
                prompt=self.send_prompt(
                    prompt=prompt,
                    human_prefix=human_prefix))
            if not prompt or prompt.lower().strip() == 'exit':
                break
        return prompt


if __name__ == '__main__':
    cli = Bot()
    cli.chat()
