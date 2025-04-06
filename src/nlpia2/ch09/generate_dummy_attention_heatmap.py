#!/usr/bin/env python
# coding: utf-8

# In[6]:


import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


ticks= ['<EOS>', 'What', 'is', 'BERT', '?', 'EOS']

correlations = np.diag([0.4, 0.8, 0.4, 0.8, 0.9, 0.4])

correlations[1][3] = 0.5 #What,BERT
correlations[1][4] = 0.5 #What, ?
correlations[3][1] = 0.5 #Bert, What
correlations[4][1] = 0.5 #?, What
correlations[3][4] = 0.3 #BERT, ?
correlations[4][3] = 0.3 #?, BERT


# In[18]:


plot = sns.heatmap(correlations, 
            xticklabels=ticks, 
            yticklabels=ticks,
           cmap=sns.cubehelix_palette(as_cmap=True))
fig= plot.get_figure()
fig.savefig('attention_heatmap.png')


# In[ ]:




