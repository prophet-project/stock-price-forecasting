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


# # Prepare dataset

# In[2]:


from src.load_datasets import load_input_dataset

df = load_input_dataset()

df.head()


# In[3]:


df.index = pd.to_datetime(df.pop('timestamp'), unit='ms')

df.head()


# In[4]:


df[::60].iplot(subplots=True)


# In[5]:


from src.prepare_datasets import add_indicators

df = add_indicators(df)


# In[12]:


df


# In[7]:


df[::60].iplot(subplots=True)


# In[11]:


size = len(df)
prediction_size = 3*31*24*60

# pycarot will split on train and test by self
data = df[0:-prediction_size]
unseen_data = df[-prediction_size:]

unseen_data


# In[14]:


from pycaret.regression import *

exp_reg_btc = setup(
    data = data, target = 'close', 
    session_id = 123, 
    log_experiment = True, log_plots=True, log_profile=True, experiment_name = 'bitcoin_close_1m',
    use_gpu = True,
    data_split_shuffle = False
)


# ### TODO
# 
# * check where MLflow store experiments data
# * try put normalized data
# * try use all features from add_ta_features and set feature_selection=True
# * try add featues by tsfreash
# * try extract features by tsfreash
# * try remove trend by sktime

# In[ ]:


best = compare_models()


# In[ ]:




