#!/usr/bin/env python
# coding: utf-8

# In[8]:


kernel = [1, 1]
inpt = [1, 2, 0, -1, -2, 0, 1, 2]

output = []
for i in range(len(inpt) - 1):  # <1>
    output.append(
        sum(
            [
                inpt[i + k] * kernel[k]
                for k in range(2)  # <2>
            ]
        )
    )


# In[9]:


# <1> Stop at second to last input value so the window size of 2 doesn't slide off the end of the input
# <2> kernel is 2 long and the list comprehension iterates over the kernel length


# In[10]:


output


# `[3, 2, -1, -3, -2, 1, 3]`

# In[11]:


def convolve(inpt, kernel):
    output = []
    for i in range(len(inpt) - len(kernel) + 1):  # <1>
        output.append(
            sum(
                [
                    inpt[i + k] * kernel[k]
                    for k in range(len(kernel))  # <2>
                ]
            )
        )
    return output


# In[12]:


# <1> Stop at second to last input value so the window size of 2 doesn't slide off the end of the input
# <2> kernel is 2 long and the list comprehension iterates over the kernel length


# In[13]:


convolve(inpt=inpt, kernel=[1, 1, 1])


# `[3, 1, -3, -3, -1, 3]`

# In[ ]:




