greetings = "Hi Hello Greetings".split()
user_statement = "Hello Joshua"
user_token_sequence = user_statement.split()
user_token_sequence
if user_token_sequence[0] in greetings:
    bot_reply = "Themonucluear War is a strange game. "  # <1>
    bot_reply += "The only winning move is NOT TO PLAY."
else:
    bot_reply = "Would you like to play a nice game of chess?"
import re  # <1>
r = "(hi|hello|hey)[ ,:.!]*([a-z]*)"  # <2>
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)  # <3>
re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)
re.match(r, "hey, what's up", flags=re.IGNORECASE)
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])(morn[gin']{0,3}|"
r += r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})")
re_greeting = re.compile(r, flags=re.IGNORECASE)  # <1>
re_greeting.match('Hello Rosa')
re_greeting.match('Hello Rosa').groups()
re_greeting.match("Good morning Rosa")
re_greeting.match("Good Manning Rosa")  # <2>
re_greeting.match('Good evening Rosa Parks').groups()  # <3>
re_greeting.match("Good Morn'n Rosa")
re_greeting.match("yo Rosa")
my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot',
    'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = ''  # <1>
match = re_greeting.match(input())
if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print("Good one.")
    elif at_name.lower() in my_names:
        print("Hi {}, How are you?".format(greeter_name))
from collections import Counter
Counter("Guten Morgen Rosa".split())
Counter("Good morning, Rosa!".split())
from itertools import permutations
[" ".join(combo) for combo in\
    permutations("Good morning Rosa!".split(), 3)]
s = """Find textbooks with titles containing 'NLP',
    or 'natural' and 'language', or
    'computational' and  'linguistics'."""
len(set(s.split()))
import numpy as np
np.arange(1, 12 + 1).prod()  # factorial(12) = arange(1, 13).prod()
