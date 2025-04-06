#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-01_Machines-that-can-read-and-write-NLP-overview`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-01_Machines-that-can-read-and-write-NLP-overview.adoc)

# #### 

# In[ ]:


import re  # <1>
r = "(hi|hello|hey)[ ,:.!]*([a-z]*)"  # <2>
re.match(r, 'Hello Rosa', flags=re.IGNORECASE)  # <3>


# #### 

# In[ ]:


re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE)


# #### 

# In[ ]:


re.match(r, "hey, what's up", flags=re.IGNORECASE)


# #### 

# In[ ]:


r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])(morn[gin']{0,3}|"
r += r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)  # <1>
re_greeting.match('Hello Rosa')


# #### 

# In[ ]:


re_greeting.match('Hello Rosa').groups()


# #### 

# In[ ]:


re_greeting.match("Good morning Rosa")


# #### 

# In[ ]:


re_greeting.match("Good Manning Rosa")  # <2>
re_greeting.match('Good evening Rosa Parks').groups()  # <3>


# #### 

# In[ ]:


re_greeting.match("Good Morn'n Rosa")


# #### 

# In[ ]:


re_greeting.match("yo Rosa")


# #### 

# In[ ]:


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


# #### 

# In[ ]:


from collections import Counter
Counter("Guten Morgen Rosa".split())


# #### 

# In[ ]:


Counter("Good morning, Rosa!".split())


# #### 

# In[ ]:


from itertools import permutations
[" ".join(combo) for combo in\
    permutations("Good morning Rosa!".split(), 3)]


# #### 

# In[ ]:


s = """Find textbooks with titles containing 'NLP',
    or 'natural' and 'language', or
    'computational' and  'linguistics'."""
len(set(s.split()))


# #### 

# In[ ]:


import numpy as np
np.arange(1, 12 + 1).prod()  # factorial(12) = arange(1, 13).prod()

