from nlpia2.chatgpt import *
from nlpia2.chatgpt import send_prompt
prompt = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
send_prompt(
    context_prompt="You are an honest and transparent virtual assistant (bot)",
    prompt=prompt)
send_prompt(
    context_prompt="You are an LLM",
    prompt=prompt)
send_prompt(
    context_prompt='',
    prompt=prompt)
send_prompt(
    context_prompt='You are a concise question answering machine learning model.',
    prompt=prompt)
prompt
prompt += " Your answer should be a number."
send_prompt(
    context_prompt='',
    prompt=prompt)
prompt = "Javonka has 5 pickleball balls. She buys 2 more packets of balls. Each packet has 4 balls. How many balls does she have now?"
prompt += " Your answer should be a single number."
prompt = "Javonka has 5 pickleball balls. She buys 2 more packets of balls. Each packet has 4 balls. How many balls does she have now?"
prompt += " Your answer should be a single number."
send_prompt(
    context_prompt='',
    prompt=prompt)
prompt = "Javonka has 5 pickleball balls. She buys 2 more packets of balls. Each packet has 4 balls. How many balls does she have now?"
prompt = f"An animal has 5 {}. She buys 2 more packets of balls. Each packet has 4 balls. How many balls does she have now?"
plural_object_name = 'ejekofa'
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers}  more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now?"
agent_first_name = 'Jekzunbe'
import random
random.randrange?
random.randint?
import numpy as np
np.random.randint?
num_objects, add_num_containers, num_objects_per_container = np.random.randint(low=0, high=30, size=3)
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers}  more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now? Your answer should be a single number NOT a sentence."
send_prompt(
    context_prompt=' ',
    prompt=prompt)
num_objects, add_num_containers, num_objects_per_container
prompt
num_objects, add_num_containers, num_objects_per_container = np.random.randint(low=0, high=30, size=3)
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers} more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now? Your answer should be a single number NOT a sentence."
context_prompt = 'You are ChatGPT-3.5-turbo, a large language model.'
num_objects, add_num_containers, num_objects_per_container
context_prompt = 'You are ChatGPT-3.5-turbo.'
prompt
send_prompt(
    context_prompt=context_prompt,
    prompt=prompt)
send_prompt(
    context_prompt=context_prompt,
    prompt=prompt)
hist -o -p -f src/nlpia2/ch10/chat_gpt_word_problem_few_shot.hist.py
hist -o -p -f src/nlpia2/ch10/chat_gpt_word_problem_few_shot.hist.ipy
hist -f src/nlpia2/ch10/chat_gpt_word_problem_few_shot.hist.ipy
hist -f src/nlpia2/ch10/chat_gpt_word_problem_few_shot.hist.py
agent_first_name = ''
agent_first_name = 'Roger'
plural_object_name = 'tennis balls'
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers} more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now? Your answer should be a single number NOT a sentence."
prompt
send_prompt(
    context_prompt=context_prompt,
    prompt=prompt)
num_objects, add_num_containers, num_objects_per_container = 5, 2, 3
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers} more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now? Your answer should be a single number NOT a sentence."
prompt

send_prompt(
    context_prompt=context_prompt,
    prompt=prompt)
prompt = f"{agent_first_name} has {num_objects} {plural_object_name}. They obtain {add_num_containers} more containers of {plural_object_name}. Each container has {num_objects_per_container} {plural_object_name}. How many {plural_object_name} do they have now? Your answer should be a single number."

send_prompt(
    context_prompt=context_prompt,
    prompt=prompt)
hist -f src/nlpia2/ch10/chat_gpt_word_problem_few_shot.hist.py
