#!/usr/bin/env python
# coding: utf-8

# In[1]:


# plotly standard imports
import plotly.graph_objs as go
import chart_studio.plotly as py

# Cufflinks wrapper on plotly
import cufflinks

# Data science imports
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot, init_notebook_mode
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Set global theme
cufflinks.set_config_file(world_readable=True, theme='pearl')


# In[2]:


import pandas as pd

df = pd.read_csv('./data/bitcoin_tweets_from_100_likes.csv')
df


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df = df[['created_at', 'full_text', 'retweet_count', 'favorite_count', 'reply_count', 'quote_count', 'lang']]
df


# In[7]:


df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')


# In[8]:


df.dropna(subset = ["created_at", "full_text"], inplace=True)


# In[9]:


df.info()


# In[10]:


df['reply_count'] = pd.to_numeric(df['reply_count'], errors='coerce')
df['quote_count'] = pd.to_numeric(df['quote_count'], errors='coerce')

df.info()


# In[11]:


df.dropna(subset = ["retweet_count", "favorite_count"], inplace=True)


# In[13]:


df.info()


# In[14]:


df['lang'].value_counts().iplot(kind='bar')


# In[15]:


df = df.loc[df['lang'] == 'en']
df = df[['created_at', 'full_text', 'retweet_count', 'favorite_count', 'reply_count', 'quote_count']]
df


# In[16]:


df.index = pd.to_datetime(df.pop('created_at'))
df


# In[17]:


df.isna().sum()


# In[19]:


sdf = df[['retweet_count', 'favorite_count', 'reply_count', 'quote_count']].groupby(pd.Grouper(freq='d')).sum()
sdf


# In[20]:


sdf = sdf.loc['2016-01-01':]
sdf


# In[33]:


sdf.iplot(subplots=True)


# In[36]:


import scipy as sc
import numpy as np

z_scores = sc.stats.zscore(sdf)
z_scores = np.abs(z_scores)
sdf_smooth = sdf[(z_scores < 2).all(axis=1)]
sdf_smooth.iplot(subplots=True)


# In[37]:


sdf_out = sdf[(z_scores >= 2).all(axis=1)]
sdf_out.iplot(subplots=True)


# In[39]:


dff = df[['retweet_count', 'favorite_count', 'reply_count', 'quote_count']].loc['2016-01-01':]

z_scores = np.abs(sc.stats.zscore(dff))
dff_smooth = dff[(z_scores < 2).all(axis=1)]

dff_smooth.groupby(pd.Grouper(freq='d')).sum().iplot(subplots=True)


# In[40]:


dff_out = dff[(z_scores >= 2).all(axis=1)]

dff_out.groupby(pd.Grouper(freq='d')).sum().iplot(subplots=True)


# In[ ]:




