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


# # Explore training metrics

# In[2]:


df = pd.read_csv('./metrics/training.csv')
df.head()


# In[3]:


df[['epoch', 'loss', 'val_loss']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='loss', 
    title='Training loss',
    linecolor='black',
)


# In[4]:


df[['epoch', 'mean_absolute_error', 'val_mean_absolute_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
)


# In[5]:


df[['epoch', 'mean_squared_logarithmic_error', 'val_mean_squared_logarithmic_error']].iplot(
    x='epoch',
    mode='lines+markers',
    xTitle='epoch',
    yTitle='mean_absolute_error', 
    title='mean_absolute_error'
)


# In[ ]:




