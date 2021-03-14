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


# # Let's explore datasets
# 
# ## input dataset
# 
# Will use target dataset Bitcoin cryptocurrency historical prices data from Binance
# 
# Bitcoin data at 1-minute intervals from April 10, 2016

# In[2]:


from src.load_datasets import load_input_dataset

df = load_input_dataset()

df.head()


# ## Convert timestamp to date index

# In[3]:


df.index = pd.to_datetime(df.pop('timestamp'), unit='ms')

df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# ### Explore correlations between diffenrences of values

# In[8]:


import seaborn as sns

corr = df[['high', 'low', 'open', 'close', 'volume']].diff().dropna().corr()

sns.heatmap(corr,cmap='Blues',annot=False) 


# In[9]:


sns.heatmap(corr, annot=True, cmap = 'viridis')


# In[13]:


import sweetviz as sv

analyse_report = sv.analyze([df, 'Bitcoin'], target_feat="close")
analyse_report.show_notebook()


# ### Feature evalution over time

# In[14]:


# traget dataset too big for plot

hour_df = df[59:: 60]

hour_df.iplot(
    subplots=True,
)


# In[15]:


df.describe().transpose()


# In[ ]:




